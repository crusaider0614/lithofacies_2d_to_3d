from pathlib import Path


def get_project_root():
    return str(Path(__file__).parent.parent)


class ValueTracker:
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
        return self.cur_value / (1 - self.bias) if self.bias < 1.0 else 0.0
