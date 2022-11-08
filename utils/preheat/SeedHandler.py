"""
    负责随机数的设定等
"""
import random
import numpy as np
import torch


class SeedHandler(object):
    __slots__ = 'seed'

    def __init__(self):
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def get_seed(self):
        return self.seed


seedHandler = SeedHandler()
