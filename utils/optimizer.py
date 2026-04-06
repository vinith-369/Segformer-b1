# ---------------------------------------------------------------
# Poly-warmup AdamW optimizer (matches official SegFormer schedule)
# ---------------------------------------------------------------

import torch


class PolyWarmupAdamW(torch.optim.AdamW):
    """AdamW with linear warmup followed by polynomial LR decay.

    Schedule:
        - Warmup phase: linearly increase LR from warmup_ratio * base_lr to base_lr
        - Poly decay phase: LR = base_lr * (1 - iter/max_iter)^power

    Args:
        warmup_iter: Number of warmup iterations.
        max_iter: Total training iterations.
        warmup_ratio: Starting LR ratio during warmup.
        power: Polynomial decay power.
    """

    def __init__(self, params, lr, weight_decay, betas,
                 warmup_iter=None, max_iter=None, warmup_ratio=None, power=None):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)

        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.max_iter = max_iter
        self.power = power
        self.__init_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        # Warmup phase
        if self.global_step < self.warmup_iter:
            lr_mult = 1 - (1 - self.global_step / self.warmup_iter) * (1 - self.warmup_ratio)
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        # Poly decay phase
        elif self.global_step < self.max_iter:
            lr_mult = (1 - self.global_step / self.max_iter) ** self.power
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__init_lr[i] * lr_mult

        super().step(closure)
        self.global_step += 1
