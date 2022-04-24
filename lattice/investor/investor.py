from abc import ABC, abstractmethod
from lattice.investor.wallet import Wallet
from lattice.investor.market import Market
from lattice.investor.order import LocalOrder
import numpy as np

class Investor:

    """
    Takes a (wallet, market) and places trades.
    All of the input objects have generic methods that generalize
    accross their class types such that investors can be written generally
    for any exchange whether it be local or online.
    """

    def __init__(self, wallet: Wallet, market: Market) -> None:
        
        self.wallet = wallet
        self.market = market
        self.orders = dict()

    def place_order(self, order: Order):
        self.wallet.update_balance(order)
        self.orders.setdefault(order.asset, []).append(order)
    
    @abstractmethod
    def evaluate_market(self):
        pass


class BernoulliInvestor(Investor):

    """
    TODO: market.place_order(params) is more generic
    """

    def __init__(self, wallet: Wallet, market: Market, **kwargs) -> None:
        super().__init__(wallet, market)
        self.__dict__.update(kwargs)

    def evaluate_market(self):
        done, prices, features = self.market.get_state() # asynchronous
        name = np.random.choice(list(prices.keys()))
        orders = [
            self.market.order(name, 'BUY', prices[name], 0.1),
            Order(name, 'SELL', prices[name], 0.1)
        ]
        new_order = np.random.choice(orders)
        self.place_order(new_order)
        return done


