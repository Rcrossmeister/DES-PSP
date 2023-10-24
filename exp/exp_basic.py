import os
import torch
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def train(self):
        pass

    def val(self):
        pass

    def test(self):
        pass
    