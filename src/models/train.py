import gc
import os

import torch as th
import torch
import numpy as np

from .. import artifacts
from .. import config
from .. import helpers
from ..data.load import load_dataset
from . import callback
from . import evaluate
from .build_model import build_model


def train(cfg, net, loader, run, eval_data, callbacks=tuple()):
    n_batches = len(loader)
    is_graph_loader = 'torch_geometric' in str(type(loader))
    print(
        f"[train] start run={run} epochs={cfg.n_epochs} batches_per_epoch={n_batches} "
        f"device={config.DEVICE} graph_loader={is_graph_loader}",
        flush=True,
    )

    if hasattr(net, 'init_centers_with_kmeans') and not getattr(net, 'kmeans_initialized_done', False):
        print(f"[train] run={run} starting spectral-clustering center init", flush=True)
        net.init_centers_with_kmeans(loader, config.DEVICE)
        net.kmeans_initialized_done = True
        print('[train] spectral clustering init done', flush=True)
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    for e in range(1, cfg.n_epochs + 1):
        net.train()
        iter_losses = []
        should_run_eval = (e % max(1, cfg.eval_interval)) == 0
        should_log_epoch_start = e == 1 or e == cfg.n_epochs or should_run_eval
        if should_log_epoch_start:
            print(f"[train] run={run} epoch {e}/{cfg.n_epochs} started", flush=True)
        for i, data in enumerate(loader):
            if is_graph_loader:
                data = data.to(config.DEVICE)
                num_views = cfg.network_config.fusion_config.n_views
                batch = [data] * num_views
            try:
                batch_losses = net.train_step(batch, epoch=(e - 1), it=i, n_batches=n_batches)
            except Exception as err:
                import traceback
                print(f'Training stopped due to exception: {err}')
                traceback.print_exc()
                return False
            iter_losses.append(helpers.npy(batch_losses))

        if should_run_eval:
            print(f"[train] run={run} epoch {e}/{cfg.n_epochs} evaluating", flush=True)
        logs = evaluate.get_logs(cfg, net, eval_data=eval_data, iter_losses=iter_losses, epoch=e, include_params=True)

        try:
            for cb in callbacks:
                cb.epoch_end(e, logs=logs, net=net)
        except callback.StopTraining as err:
            print(err)
            break

    for cb in callbacks:
        if isinstance(cb, callback.ModelSaver):
            best_model_path = cb.save_dir / 'best.pt'
            if not best_model_path.exists():
                cb._save_model('best.pt', net)
                print('Training ended, saved final model as best.pt')

    return True


def main():
    experiment_name, cfg, note = config.get_experiment_config()
    random_mode = getattr(cfg, 'random', False)
    print(
        f"[main] config={experiment_name} dataset={cfg.dataset_config.name} seed={cfg.seed} "
        f"random={random_mode} epochs={cfg.n_epochs} runs={cfg.n_runs}",
        flush=True,
    )
    helpers.set_seed(cfg.seed, random_mode=random_mode)

    experiment_identifier = artifacts.timestamp_now()
    experiment_dir = artifacts.get_training_experiment_dir(
        dataset_name=cfg.dataset_config.name,
        experiment_name=experiment_name,
        identifier=experiment_identifier,
        category='main',
    )
    artifacts.write_metadata(
        experiment_dir,
        {
            'experiment_type': 'main',
            'dataset': cfg.dataset_config.name,
            'config_name': experiment_name,
            'identifier': experiment_identifier,
            'seed': int(cfg.seed),
            'random': bool(random_mode),
            'n_epochs': int(cfg.n_epochs),
            'n_runs': int(cfg.n_runs),
            'note': note,
            'fusion_rule': 'multi_source_pcr6',
        },
    )
    print(f"[main] experiment_dir={experiment_dir}", flush=True)

    print(f"[main] loading dataset", flush=True)
    dataset = load_dataset(**cfg.dataset_config.model_dump())
    print(f"[main] dataset ready size={len(dataset)}", flush=True)

    from torch_geometric.loader import DataLoader as PyGDataLoader
    from src.data.load import collate_fn_with_laplacian
    from src.lib.norm_ops import apply_gn_backbone_post

    loader_generator = helpers.make_torch_generator(cfg.seed, random_mode=random_mode)
    worker_init_fn = helpers.make_worker_init_fn(cfg.seed, random_mode=random_mode)
    use_workers = int(os.getenv("CUE_NUM_WORKERS", "4"))
    use_workers = max(0, use_workers)
    print(f"[main] building train loader workers={use_workers} batch_size={int(cfg.batch_size)}", flush=True)
    loader = PyGDataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=use_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=collate_fn_with_laplacian,
        persistent_workers=bool(use_workers > 0),
        generator=loader_generator,
        worker_init_fn=worker_init_fn,
    )
    print(f"[main] train loader ready batches={len(loader)}", flush=True)

    print("[main] building evaluation loader", flush=True)
    eval_data = evaluate.get_eval_data(
        dataset,
        cfg.n_eval_samples,
        cfg.batch_size,
        seed=cfg.seed,
        random_mode=random_mode,
    )
    print("[main] evaluation loader ready", flush=True)

    print(cfg.seed)
    run_logs = []
    print(f"\n{'=' * 60}")
    print(f'Begin training: {cfg.n_runs} runs x {cfg.n_epochs} epochs')
    if note:
        print(f'Note: {note}')
    print(f"{'=' * 60}\n")

    for run in range(cfg.n_runs):
        print(f"\n>>> Run {run + 1}/{cfg.n_runs} start <<<")
        print(f"[main] run={run} building model", flush=True)
        net = build_model(cfg.network_config)
        apply_gn_backbone_post(net, gn_groups=8, verbose=False)
        run_dir = artifacts.get_run_dir(experiment_dir, run)
        print(f"[main] run={run} artifact_dir={run_dir}", flush=True)

        callbacks = (
            callback.Printer(
                print_confusion_matrix=(cfg.network_config.cm_config.n_clusters <= 100),
                verbose=False,
            ),
            callback.ModelSaver(
                cfg=cfg,
                experiment_name=experiment_name,
                identifier=experiment_identifier,
                run=run,
                epoch_interval=1,
                checkpoint_interval=cfg.checkpoint_interval,
                save_dir=run_dir,
            ),
            callback.EarlyStopping(
                patience=cfg.patience,
                best_loss_term=cfg.best_loss_term,
                epoch_interval=1,
            ),
        )
        train(cfg, net, loader, run, eval_data=eval_data, callbacks=callbacks)
        final_logs = evaluate.eval_run(
            cfg=cfg,
            cfg_name=experiment_name,
            experiment_identifier=experiment_identifier,
            run=run,
            net=net,
            eval_data=eval_data,
            callbacks=callbacks,
            load_best=True,
        )
        run_logs.append(final_logs)
        acc = final_logs.get("metrics/acc", float("nan"))
        nmi = final_logs.get("metrics/nmi", float("nan"))
        ari = final_logs.get("metrics/ari", float("nan"))
        print(
            f"\n[main] run={run} best-model metrics | "
            f"ACC={acc:.4f} | NMI={nmi:.4f} | ARI={ari:.4f}",
            flush=True,
        )


if __name__ == '__main__':
    main()
