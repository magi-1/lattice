from lattice.wallet import *
from lattice.market import *
from lattice.order import *
from lattice.broker import *

from abc import ABC, abstractmethod
import numpy as np
from typing import List


class Investor:

    def __init__(
        self, 
        wallet: Wallet, 
        market: Market, 
        broker: Broker
    ) -> None:
        self.wallet = wallet
        self.market = market
        self.broker = broker

    def submit_orders(self, orders: List[Order]) -> None:
        for order in orders:
            if self.wallet.can_afford(order):
                self.broker.place_order(order)
                self.wallet.update_balance(order)
            
    def cancel_orders(self, order_ids: List[str]) -> None:
        for oid in order_ids:
            self.broker.cancel_order(oid)
    
    @abstractmethod
    def evaluate_market(self):
        pass


class BernoulliInvestor(Investor):

    def __init__(
        self, 
        wallet: Wallet, 
        market: Market, 
        broker: Broker,
        p: list
    ) -> None:
        super().__init__(wallet, market, broker)
        self.p = p
        
    def evaluate_market(self) -> bool:
        
        # Check state of the market
        done, time, prices, features = self.market.get_state() 
        self.wallet.update_total_value(prices)

        # Select an action / make a decisions
        asset_name = np.random.choice(list(prices.keys()))

        # Create an order
        order = self.broker.create_order(
            asset=asset_name,
            side=np.random.choice(['BUY','SELL'], p = self.p),
            size=0.01,
            open_price=prices[asset_name],
            open_time=time
            )
        self.submit_orders([order])

        if not done:
            return self.wallet.total_value