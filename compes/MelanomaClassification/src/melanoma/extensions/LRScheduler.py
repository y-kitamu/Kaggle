import math
import os

import chainer


class WarmUpCosineAnnealing(chainer.training.Extension):
    """Warm up and Cyclic Learning Rate Scheduler.
    When each cycle ends, serialize model weights.

    Args:
         attr (string)                : scheuled parameter name
         value_range (tuple of float) : The first and the last values of the attribute.
         init_cycle_itr (int)         : initial cycle period (unit : iteration)
         cycle_scale (int)            : scaling next cycle period
         attr_scale (int)             : Log scale factor of the first value of the attribute
    """

    def __init__(self,
                 attr,
                 value_range=(1e-3, 1e-5),
                 warm_up_itr=400,
                 init_cycle_itr=800,
                 cycle_scale=2,
                 attr_scale=1.0):
        self._attr = attr
        self._in_warming_up = True
        self._warm_up_itr = warm_up_itr
        self._cycle_itr = init_cycle_itr
        self._cycle_scale = cycle_scale
        self._log_max = math.log(value_range[0])
        self._log_min = math.log(value_range[1])
        self._attr_scale = attr_scale
        self._t = 0
        self._count = 0

    def __call__(self, trainer):
        self._t += 1
        optimizer = trainer.updater.get_optimizer("main")
        value = self._get_next_value(trainer)
        setattr(optimizer, self._attr, value)

    def _get_next_value(self, trainer):
        if self._in_warming_up:
            if self._t < self._warm_up_itr:
                return math.exp(self._log_min)
            self._in_warming_up = False
            self._t = 1

        if self._t > self._cycle_itr:
            self._serialize_model(trainer)
            self._t = 1
            self._count += 1
            self._cycle_itr *= self.cycle_scale
            self._log_max *= self._attr_scale
            self._log_min *= self._attr_scale

        return math.exp((self._log_max - self._log_min) * math.cos(math.pi * self._t / self._cycle_itr) +
                        (self._log_max + self._log_min) / 2.0)

    def _serialize_model(self, trainer):
        file_path = os.path.join(trainer.out, f"snapshot_model_lr{self._count:02d}.npz")
        print(f"save {file_path}")
        chainer.serializers.save_npz(file_path, trainer.updater.get_optimizer("main").target)
