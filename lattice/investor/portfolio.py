from typing import List, NewType,
from dataclasses import dataclass


"""
list with < 100 integers

"""

"""
RemoteStrategy vs LocalStrategy inherit the same objects but for live trading vs backtesting
"""

@dataclass
class Asset:
    long_amt: float
    short_amt: float


class Portfolio:

    def __init__(self, capital: float) -> None:
        self.capital = capital

    def calculate_something(self):
        pass

from abc import ABC, abstract_method

class AbstractOrder(ABC):
    
    def __init__(self):
        self.open_time = 0
        self.close_time = 0
        self.open_price = 0
        self.close_price = 0
        self.profit = 0

    @abstract_method
    def place(self):
        pass

    @abstract_method
    def modify(self):
        pass

    def status(self) -> dict: # vaugue
        pass

    def cancel(self, order_id: int) -> None:
        pass

class Investor:

    """
    For the live version, need an orchestrator to populate all relevant objects with data.
    For example, FTXPortfolio would be a valid input to RemoteInvestor
    """

    def __init__(self, portfolio: Portfolio=None) -> None:
        self.portfolio = portfolio
        self.orders = self._update_orders(portfolio)

    def _update_orders(self, portfolio: Portfolio) -> Dict[str,Order]:
        pass

    def open(self, order: Order) -> REsp
        pass

    def close(self, order: Order):
        pass

class Universe:
    """
    A collection of asset types where the assets can have additional meta data 
    used as contraints, hyperparameters, etc
    """
    pass

class Strategy:
    pass