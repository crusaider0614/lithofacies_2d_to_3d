"""Project-level helpers: repository root path and a running-loss tracker."""
from pathlib import Path


def get_project_root():
    """Absolute path of the repository root; all data/checkpoint/config paths are built from it."""
    return str(Path(__file__).parent.parent)


class ValueTracker:
    """Exponential moving average with bias correction (as in Adam), used to smooth the training loss."""

    def __init__(self, ema_coeff):
        self.ema_coeff = ema_coeff
        self.cur_value = 0.0
        self.bias = 1.0

    def initialize(self):
        self.__init__(self.ema_coeff)

    def feed(self, value):
        self.cur_value = self.ema_coeff * self.cur_value + (1.0 - self.ema_coeff) * value
        self.bias = self.ema_coeff * self.bias

    def val(self):
        # Divide out the bias toward the zero initial value; 0.0 before the first feed().
        return self.cur_value / (1 - self.bias) if self.bias < 1.0 else 0.0
