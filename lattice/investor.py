from lattice.wallet import *
from lattice.market import *
from lattice.broker import *
from lattice.order import *
from lattice.utils import logging
from lattice.config import InvestorConfig

from abc import ABC, abstractmethod
from pathlib import Path
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

    def evaluate_market(self) -> bool:

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

        if not done:
            return True


class GNNInvestor(Investor):
    def __init__(self, wallet, market, broker, config, train=False) -> None:
        super().__init__(wallet, market, broker, config)
        self.weight_dir = paths.weights / self.name
        self.network = hk.without_apply_rng(hk.transform(gnn.network_definition))
        self.train = train
        self.initialized = False
        self.action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}

        if self.train:
            self.experience = logging.ExperienceBuffer()

    def set_params(self, graph: jraph.GraphsTuple) -> None:
        if Path(self.weight_dir).exists():
            self.params = jax.numpy.load(self.weight_dir)
        else:
            self.params = network.init(jax.random.PRNGKey(self.seed), graph)
            jax.numpy.save(self.weight_dir, self.params)
        self.initialized = True

    def get_actions(self, features, global_features) -> List[str]:
        graph = gnn.construct_graph(features=features, global_features=global_features)
        self.initialize_params(graph)
        logits = self.network.apply(self.params, graph)
        actions = jax.random.categorical(
            key=jax.random.PRNGKey(self.seed), logits=logits
        )
        return [self.action_map[index] for index in actions]

    def evaluate_market(self) -> Union[bool, logging.ExperienceBuffer]:
        # Check state of the market
        done, time, prices, market_features = self.market.get_state()

        # Calling GNN model
        outputs = self.get_actions(
            features=market_features, global_features=self.wallet.balances.values()
        )

        # Creating orders
        orders = [None]
        for i, market_name in enumerate(self.market.markets):
            action = outputs[i]
            if action == "BUY":
                order = self.broker.market_order(
                    market=market_name,
                    side="BUY",
                    size=0.01,
                    open_price=prices[market_name],
                    open_time=time,
                )
            elif action == "SELL":
                order = self.broker.market_order(
                    market=market_name,
                    side="SELL",
                    size=0.01,
                    open_price=prices[market_name],
                    open_time=time,
                )
            orders.append(order)
        self.submit_orders([order], prices)

        if self.train:
            self.experience.push(time=time, graph=graph, outputs=outputs)

        if not done and self.train:
            r = self.wallet.get_history()["total_value"].values
            self.experience.set_reward(r)
            return self.experience

        return True
