from lattice.order import Order
from lattice.utils.io import wallet_config
from typing import List, Union, Dict
from abc import ABC, abstractmethod
import pandas as pd


class Wallet(ABC):

    def __init__(self, config):
        self.__dict__.update(config)
        self.total_value = 0
        self.history = []
    
    def can_afford(self, order: Union[Order, None]):
        if order:
            asset, underlying = order.components()
            if order.side == 'BUY' and order.value < self.balances[underlying]:
                    return True
            elif order.side == 'SELL' and asset in self.balances:
                if order.size < self.balances[asset]:
                    return True
        return False

    def update_total_value(self, prices: Dict[str,float]):
        usd = self.balances['USD']
        other = 0
        for asset,amount in self.balances.items():
            if asset != 'USD':
                other += prices[asset+'_USD']*amount
        self.total_value = other + usd            

    def update_balance(self, order: Order):
        asset, underlying = order.components()
        if asset in self.balances:
            self.balances[asset] += order.amount
        else:
            self.balances.setdefault(asset, order.amount)
        self.balances[underlying] -= order.value
        stamped_data = {**self.balances, **{'time': order.open_time}}
        self.history.append(stamped_data)
            
    def get_history(self):
        return pd.DataFrame(self.history)


@wallet_config
class LocalWallet(Wallet):
    
    def __init__(self, config):
        super().__init__(config)
    

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