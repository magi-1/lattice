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

    def submit_orders(
        self, 
        orders: List[Order], 
        prices: Dict[str,float]
    ) -> None:
        for order in orders:
            if self.wallet.can_afford(order):
                self.broker.place_order(order)
                self.wallet.update_balance(order, prices)
            
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
        self.hourly_limit = int(60/5)
        
    def evaluate_market(self):
        
        # Check state of the market
        done, time, prices, features = self.market.get_state() 

        # Select an action / make a decisions
        asset_name = np.random.choice(list(prices.keys()))

        # Create an order
        if self.market.time%self.hourly_limit==0:
            order = self.broker.create_order(
                asset=asset_name,
                side=np.random.choice(['BUY','SELL'], p = self.p),
                size=0.01,
                open_price=prices[asset_name],
                open_time=time
                )
        else:
            order = None
        self.submit_orders([order], prices)

        # Expose data 
        if not done:
            return self.wallet.total_value