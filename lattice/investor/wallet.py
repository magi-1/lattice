from lattice.order import Order

from typing import List, NewType
from abc import ABC, abstractmethod


class Wallet(ABC):
    
    @property
    def liquid_value(self):
        """
        NOTE: Might actually be better to store the number of coins
              such that to get the true value at any point in time we can pass
              in a price dictionary.
        """
        usd = self.balances['USD']
        other = sum([amt for asset,amt in self.balances.items() if asset!='USD'])
        return usd-other

    def can_afford(self, order: Order):
        pass


class LocalWallet(Wallet):
    
    def __init__(self, config):
        self.balances = config['balances']
        self.history = []
    
    def update_balance(self, order):
        asset, underlying = order.asset.split('_')
        if asset in self.balances:
            self.balances[asset] += order.value
        else:
            self.balances.setdefault(asset, order.value)
        self.balances[underlying] -= order.value
        self.history.append(self.value)
            

"""
class FTXWallet(Wallet):
    
    # Gets initialized with FTX wallet data.

    def __init__(self):
        pass

    def load_config(self):
        pass

    def get_balances(self):
        # https://docs.ftx.us/#get-balances
        pass
"""