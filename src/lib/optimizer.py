import torch as th

class Optimizer:
    def __init__(self, cfg, params):
        self.clip_norm = cfg.clip_norm
        self.params = params
        self._opt = th.optim.Adam(params, lr=cfg.learning_rate)
        if cfg.scheduler_step_size is not None:
            assert cfg.scheduler_gamma is not None
            self._sch = th.optim.lr_scheduler.StepLR(self._opt, step_size=cfg.scheduler_step_size,
                                                     gamma=cfg.scheduler_gamma)
        else:
            self._sch = None

    def zero_grad(self):
        return self._opt.zero_grad()

    def step(self, epoch):
        if self._sch is not None:
            if epoch.is_integer() and epoch > 0:
                self._sch.step()

        if self.clip_norm is not None:
            # 抑制warning（empty generator是预期行为）
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*parameters.*empty generator.*')
                th.nn.utils.clip_grad_norm_(self.params, self.clip_norm)

        out = self._opt.step()
        return out
