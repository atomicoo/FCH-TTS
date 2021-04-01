from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    """Set adam lr to the max desired lr - it will be the peak lr"""

    def __init__(self, optimizer, warmup_steps, init_scale, min=0):
        """
        true_lr = init_scale * lr in first step

        :param optimizer:
        :param warmup_steps:
        :param init_scale:
        :param min: minimum value of lr
        """
        self.warmup_steps = warmup_steps
        self.min = min
        self.init_scale = init_scale
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = self.last_epoch

        if last_epoch <= self.warmup_steps:
            c = (1-self.init_scale)/self.warmup_steps
            scale = c * last_epoch + self.init_scale
        else:
            scale = max(1, self.warmup_steps) ** 0.5 * last_epoch ** (-0.5)

        #scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [max(base_lr * scale, self.min) for base_lr in self.base_lrs]


def get_lr(last_epoch, warmup_steps, init_scale):
    if last_epoch <= warmup_steps:
        c = (1 - init_scale) / warmup_steps
        scale = c * last_epoch + init_scale
    else:
        scale = max(1, warmup_steps) ** 0.5 * last_epoch ** (-0.5)
    return scale

"""
import matplotlib.pyplot as plt
x = [0.001*get_lr(i, 0, 0) for i in range(1,250)]
print(x[200])
plt.plot(x)
plt.show()
"""


