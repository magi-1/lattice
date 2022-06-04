from lattice.wallet import *
from lattice.market import *
from lattice.broker import *
from lattice.order import *
from lattice.config import InvestorConfig

from abc import ABC, abstractmethod
from typing import List
import numpy as np


def get_investor(wallet, market, broker, config):
    name = config["class"]
    for cls in Investor.__subclasses__():
        if cls.__name__ == name:
            return cls(wallet, market, broker, config)
    raise ValueError(f"There is no Investor subclass called {name}")


class Investor:
    def __init__(
        self, wallet: Wallet, market: Market, broker: Broker, config: InvestorConfig
    ) -> None:
        self.__dict__.update(config)
        self.wallet = wallet
        self.market = market
        self.broker = broker

    def submit_orders(self, orders: List[Order], prices: Dict[str, float]) -> None:
        for order in orders:
            if self.wallet.can_afford(order):
                success = self.broker.place_order(order)
                if success:
                    self.wallet.update_balance(order, prices)

    def cancel_orders(self, order_ids: List[str]) -> None:
        for oid in order_ids:
            self.broker.cancel_order(oid)

    @abstractmethod
    def evaluate_market(self):
        pass


class BernoulliInvestor(Investor):
    def __init__(self, wallet, market, broker, config) -> None:
        super().__init__(wallet, market, broker, config)
        self.hourly_limit = int(60 / 5)

    def evaluate_market(self):

        # Check state of the market
        done, time, prices, features = self.market.get_state()

        # Select an action / make a decisions
        market_name = np.random.choice(self.market.markets)

        # Create an order
        if self.market.t % self.hourly_limit == 0:
            order = self.broker.market_order(
                market=market_name,
                side=np.random.choice(["BUY", "SELL"], p=self.p),
                size=0.01,
                open_price=prices[market_name],
                open_time=time,
            )
        else:
            order = None
        self.submit_orders([order], prices)

        # Expose data
        if not done:
            return self.wallet.total_value


class GNNInvestor(Investor):
    def __init__(self, wallet, market, broker, config) -> None:
        super().__init__(wallet, market, broker, config)

    def evaluate_market(self):

        # Check state of the market
        done, time, prices, market_features = self.market.get_state()
        graph = gnn.construct_graph(
            features=market_features, global_features=self.wallet.balances
        )

        output = self.model(market_features, global_features)

        orders = [None]
        for market_name in self.market.markets:
            action = output[market_name]
            if action:
                order = self.broker.market_order(
                    market=market_name,
                    side="BUY",
                    size=0.01,
                    open_price=prices[market_name],
                    open_time=time,
                )
            else:
                order = self.broker.market_order(
                    market=market_name,
                    side="BUY",
                    size=0.01,
                    open_price=prices[market_name],
                    open_time=time,
                )
            orders.append(order)

        self.submit_orders([order], prices)

        # Expose data (needs to be SAR data for RL purposes later down the line :D)
        if not done:
            # need to return all the features, actions, and total wallet value!!
            return self.wallet.total_value
