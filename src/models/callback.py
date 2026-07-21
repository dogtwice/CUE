import os
import yaml
import pickle
import numpy as np
import torch as th
import subprocess
from tabulate import tabulate

from .. import helpers



class Callback:
    def __init__(self, epoch_interval=1, batch_interval=1):
        self.epoch_interval = epoch_interval
        self.batch_interval = batch_interval

    def epoch_end(self, epoch, **kwargs):
        if not (epoch % self.epoch_interval):
            return self.at_epoch_end(epoch, **kwargs)

    def at_epoch_end(self, epoch, logs=None, net=None, **kwargs):
        pass

    def at_eval(self, net=None, logs=None, **kwargs):
        pass


class Printer(Callback):
    def __init__(self, print_confusion_matrix=True, verbose=True, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose  # 控制是否详细输出
        self.ignore_keys = ["iter_losses/"]
        if not print_confusion_matrix:
            self.ignore_keys.append("metrics/cmat")
            
        np.set_printoptions(edgeitems=20, linewidth=200)

    def at_epoch_end(self, epoch, logs=None, net=None, **kwargs):
        if not self.verbose:
            # 简洁模式：只打印关键指标
            key_metrics = {}
            if 'metrics/acc' in logs:
                key_metrics['ACC'] = f"{logs['metrics/acc']:.4f}"
            if 'metrics/nmi' in logs:
                key_metrics['NMI'] = f"{logs['metrics/nmi']:.4f}"
            if 'metrics/ari' in logs:
                key_metrics['ARI'] = f"{logs['metrics/ari']:.4f}"
            if 'eval_losses/total_loss' in logs:
                key_metrics['Loss'] = f"{logs['eval_losses/total_loss']:.4f}"
            
            metrics_str = ' | '.join([f"{k}: {v}" for k, v in key_metrics.items()])
            print(f"Epoch {epoch:3d} | {metrics_str}", flush=True)
            
            # 在主指标之后打印聚类指标
            clustering_metrics = {}
            if 'metrics/kmeans_acc' in logs:
                clustering_metrics['KMeans-ACC'] = f"{logs['metrics/kmeans_acc']:.4f}"
            if 'metrics/kmeans_nmi' in logs:
                clustering_metrics['KMeans-NMI'] = f"{logs['metrics/kmeans_nmi']:.4f}"
            if 'metrics/kmeans_ari' in logs:
                clustering_metrics['KMeans-ARI'] = f"{logs['metrics/kmeans_ari']:.4f}"
            
            if clustering_metrics:
                clustering_str = ' | '.join([f"{k}: {v}" for k, v in clustering_metrics.items()])
                print(f"           [KMeans] {clustering_str}", flush=True)
            
            spectral_metrics = {}
            if 'metrics/spectral_acc' in logs:
                spectral_metrics['Spectral-ACC'] = f"{logs['metrics/spectral_acc']:.4f}"
            if 'metrics/spectral_nmi' in logs:
                spectral_metrics['Spectral-NMI'] = f"{logs['metrics/spectral_nmi']:.4f}"
            if 'metrics/spectral_ari' in logs:
                spectral_metrics['Spectral-ARI'] = f"{logs['metrics/spectral_ari']:.4f}"
            
            if spectral_metrics:
                spectral_str = ' | '.join([f"{k}: {v}" for k, v in spectral_metrics.items()])
                print(f"           [Spectral] {spectral_str}", flush=True)
            
            return
        
        # 详细模式：打印所有指标
        print_logs = logs.copy()
        for key in logs.keys():
            if any([key.startswith(ik) for ik in self.ignore_keys]):
                del print_logs[key]

        # 转换numpy数组为标量或格式化字符串
        for key, value in print_logs.items():
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    print_logs[key] = value.item()
                else:
                    # 多元素数组转为字符串
                    print_logs[key] = np.array2string(value, precision=4, separator=', ')
        
        headers = ["Key", "Value"]
        values = list(print_logs.items())
        print(tabulate(values, headers=headers), "\n")


class ModelSaver(Callback):
    def __init__(self, cfg, experiment_name, identifier, run, checkpoint_interval=1, save_dir=None, **kwargs):
        super().__init__(**kwargs)

        self.max_acc = -np.inf  # 跟踪最大准确率
        self.checkpoint_interval = checkpoint_interval
        self.save_dir = save_dir or helpers.get_save_dir(
            experiment_name,
            identifier,
            run,
            dataset_name=cfg.dataset_config.name,
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self._save_cfg(cfg)

    def _save_cfg(self, cfg):
        with open(self.save_dir / "config.yml", "w") as f:
            yaml.dump(cfg.dict(), f)
        with open(self.save_dir / "config.pkl", "wb") as f:
            pickle.dump(cfg, f)

    def _save_model(self, file_name, net):
        model_path = self.save_dir / file_name
        th.save(net.state_dict(), model_path)
        print(f"Model successfully saved: {model_path}")

    def at_epoch_end(self, epoch, outputs=None, logs=None, net=None, **kwargs):
        if not (epoch % self.checkpoint_interval):
            # Save model checkpoint
            self._save_model(f"checkpoint_{str(epoch).zfill(4)}.pt", net)

        # 根据准确率最大值保存最佳模型
        acc = logs.get("metrics/acc", -np.inf)
        if acc > self.max_acc:
            self.max_acc = acc
            self._save_model("best.pt", net)
            if epoch > 0:  # 避免第一个epoch打印信息
                print(f"New best accuracy: {acc:.4f}")


class StopTraining(Exception):
    pass


class EarlyStopping(Callback):
    def __init__(self, patience, best_loss_term, **kwargs):
        super().__init__(**kwargs)
        self.best_loss_term = f"eval_losses/{best_loss_term}"
        self.patience = patience
        self.min_loss = np.inf
        self.best_epoch = 0

    def at_epoch_end(self, epoch, outputs=None, logs=None, net=None, **kwargs):
        avg_loss = logs.get(self.best_loss_term, np.inf)

        if np.isnan(avg_loss):
            raise StopTraining(f"Got loss = NaN. Training stopped.")

        if avg_loss < self.min_loss:
            self.min_loss = avg_loss
            self.best_epoch = epoch

        if (epoch - self.best_epoch) >= self.patience:
            raise StopTraining(f"Loss has not decreased in {self.patience} epochs. Min. loss was {self.min_loss}. "
                               f"Training stopped.")

