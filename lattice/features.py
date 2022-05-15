import functools
import numpy as np
from typing import NewType


"""
Class registry tooling
"""

feature_registry = {}

def register(cls):
    feature_registry[cls.__name__] = cls
    #@functools.wraps(cls)
    #def wrapper(*args, **kwargs):
    #    new_class = cls(*args, **kwargs)
    #    return new_class
    return cls


"""
Custom hyperparameter types
"""

PositiveScalar = NewType('PositiveScalar', float)


"""
Base classes
"""

class MarketFeature:
    
    def __init__(self, params):
        self.__dict__.update(params)

    def evaluate(self, df):
        pass

class OrderBookFeature:
    
    def __init__(self, params):
        self.__dict__.update(params)

    def evaluate(self, df):
        pass


"""
Market features
"""

@register
class EMA(MarketFeature):
    
    # alpha: float

    def __init__(self, params):
        super().__init__(params)
        self.alpha = float(self.alpha)
    
    def evaluate(self, prices: np.array, _volumes: np.array):
        values = []
        if hasattr(self, 'prev'):
            for i in range(1, len(prices)):
                x = self.alpha*prices[i] + (1-self.alpha)*self.prev
                self.prev = x
                values.append(x)
        else:
            self.prev = prices[0]
        return np.array(values)