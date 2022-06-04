import functools
import numpy as np
from typing import NewType, TypeVar, Dict, Union


"""
Class registry tooling
"""

feature_registry = {}


def register(cls):
    feature_registry[cls.__name__] = cls
    return cls


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
Custom feature and hyperparameter types
"""

PositiveScalar = NewType("PositiveScalar", float)


class NodeFeature:
    pass


class EdgeFeature:
    pass


FeatureDict = Dict[str, Union[NodeFeature, EdgeFeature]]

"""
Market features
"""


@register
class EMA(MarketFeature, NodeFeature):
    def __init__(self, params):
        super().__init__(params)
        self.alpha = float(self.alpha)

    def evaluate(self, prices: np.array, _volumes: np.array):
        values = []
        if not hasattr(self, "prev"):
            self.prev = prices[0]
        for i in range(1, len(prices)):
            x = self.alpha * prices[i] + (1 - self.alpha) * self.prev
            self.prev = x
            values.append(x)
        return np.array(values)


@register
class Volatility(MarketFeature, NodeFeature):
    def __init__(self, params):
        super().__init__(params)

    def evaluate(self, prices: np.array, _volumes: np.array):
        return np.std(prices, axis=0)


@register
class LogReturns(MarketFeature, NodeFeature):
    def __init__(self, params):
        super().__init__(params)

    def evaluate(self, prices: np.array, _volumes: np.array):
        return np.diff(np.log(prices), axis=0)
