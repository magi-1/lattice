from typing import List, NewType
from abc import ABC, abstractmethod


class Wallet(ABC):
    pass


class FTXWallet(Wallet):
    
    """
    Gets initialized with FTX wallet data.
    """

    def __init__(self):
        pass

    def load_config(self):
        pass

    def get_balances(self):
        # https://docs.ftx.us/#get-balances
        pass


class LocalWallet(Wallet):
    
    def __init__(self, config):
        self.__dict__.update(config['wallet'])
        self.history = []
    
    def update_balance(self, order):
        asset, underlying = order.asset.split('_')
        if asset in self.balances:
            self.balances[asset] += order.value
        else:
            self.balances.setdefault(asset, order.value)
        self.balances[underlying] -= order.value
        self.history.append(self.value)
            
    @property
    def value(self):
        usd = self.balances['USD']
        other = sum([amt for asset,amt in self.balances.items() if asset!='USD'])
        return usd-other