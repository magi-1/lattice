from lattice.config import WalletConfig
from lattice.order import Order
from lattice.clients import FtxClient

from typing import List, Union, Dict
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import pandas as pd
import copy
import os

load_dotenv()


class Wallet(ABC):
    def __init__(self, config: WalletConfig):
        self.__dict__.update(config)
        self.default_balances = self.balances
        self.total_value = 0.01
        self.history = []

    def reset(self):
        self.balances = copy.deepcopy(self.default_balances) 
        self.total_value = 0.01
        self.history = []

    def can_afford(self, order: Union[Order, None]):
        if order:
            asset, underlying = order.components()
            if order.side == "BUY" and order.value < self.balances[underlying]:
                return True
            elif order.side == "SELL" and asset in self.balances:
                if order.size < self.balances[asset]:
                    return True
        return False

    def update_balance(self, order: Order, prices: Dict[str, float]) -> None:

        # Updating asset quantities
        asset, underlying = order.components()
        if asset in self.balances:
            self.balances[asset] += order.amount
        else:
            self.balances.setdefault(asset, order.amount)

        # Add/deduct from underling
        fee = 1 + order.fee if order.side == "BUY" else 1
        self.balances[underlying] -= order.value * fee

        # Getting total portfolio value
        total_value = self.balances["USD"]
        for asset, amount in self.balances.items():
            if asset != "USD":
                total_value += prices[asset + "_USD"] * amount
        self.total_value = total_value

        # Logging wallet state
        derived_data = {"total_value": total_value}
        meta_data = {"time": order.open_time}
        stamped_data = {
            **meta_data,
            **derived_data,
            **self.balances,
        }
        self.history.append(stamped_data)

    def get_history(self):
        return pd.DataFrame(self.history)

    def featurize(self):
        # TODO: For GNN global feature
        pass


class LocalWallet(Wallet):
    def __init__(self, config):
        super().__init__(config)


class FTXWallet(Wallet):
    def __init__(self, config):
        super().__init__(config)
        client = FtxClient(
            api_key=os.environ["FTX_DATA_KEY"], api_secret=os.environ["FTX_DATA_SECRET"]
        )
        self.balances = self.pull_balances()

    def pull_balances(self):
        balances = {}
        for data in self.client.get_balances():
            balances[data["coin"]] = data["free"]
        return balances
