from lattice.investor.wallet import Wallet
from lattice.investor.market import Market
from lattice.investor.order import LocalOrder

from abc import ABC, abstractmethod
import numpy as np
from typing import List


class Investor:

    def __init__(self, wallet: Wallet, market: Market, orders: Orders) -> None:
        self.wallet = wallet
        self.market = market
        self.orders = orders

    def submit_orders(self, orders: List[Orders]) -> None:
        for order in orders:
            self.broker.place_order(order)
            self.wallet.update_balance(order)
            
    def cancel_orders(self, order_ids: List[str]) -> None:
        for oid in order_ids:
            self.broker.cancel_order(oid)
    
    @abstractmethod
    def evaluate_market(self):
        pass


class BernoulliInvestor(Investor):

    def __init__(self, wallet: Wallet, market: Market, **kwargs) -> None:
        super().__init__(wallet, market)
        self.__dict__.update(kwargs)
        
    def evaluate_market(self):
        
        # Check state of the market
        done, prices, features = self.market.get_state() 

        # Select an action / make a decisions
        asset_name = np.random.choice(list(prices.keys()))

        # Create an order
        order = self.broker.create_order(
            asset=asset_name,
            side='buy',
            size=0.1,
            open_price=prices[asset_name]
            )

        self.submit_orders([order])
       
