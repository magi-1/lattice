from abc import ABC, abstractmethod
from lattice.investor.wallet import Wallet
import pandas as pd
import numpy as np
import datetime
import lattice.paths as paths

"""
All markets take the same kind of config then know
how to interact with the real/synthetic market by pulling the
data down at a given rate or only piping out a subset of tickers. 

Later on these can also take config['market']['features'] = List[str]
and call feature classes to pipe out the data. Can also let the market have
a buffer mechanism if desired so that recurrent data can be used for a given model.

^ This also falls inline with having methods to extract and process orderbook information.
"""

class Market(ABC):

    """
    - Takes as input a market config specifying either
      the whole market or a subset of the market
    - Handles the data streams / asynchronous code
    """

    #def __init__(self, config):
    #    self.dataset = config['dataset']


class LocalMarket(Market):
   
    def __init__(self, config) -> None:
        #super().__init__(config)
        self.dataset = config['dataset']
        self.assets = config['assets']
        self.data = self._load_data()
        self.T = self.data['BTC_USD'].shape[0]
        self.time = 0
        
    def to_timestamp(self, iso_time: str):
        return datetime.datetime.fromisoformat(iso_time).timestamp()*1000
        
    def _load_data(self):
        t0,t1 = list(map(self.to_timestamp, self.window))
        data_dir = paths.data/self.dataset
        data = dict()
        for path in data_dir.iterdir():
            asset_name = path.name.split('.')[0]
            if asset_name in self.assets:
                df = pd.read_parquet(path).query('@t0 <= time < @t1')
                # TODO: Add more features
                df.loc[:,'log_ret'] = np.log(df.close.values) - np.log(df.close.shift(1).values)
                data[asset_name] = df.iloc[1:]
        return data
    
    def get_state(self):
        prices, features = dict(), []
        for name in self.assets:
            # TODO: Add feature class that can process
            # lagging snapshots to have full flexiblity
            # in terms of feature design
            x = self.data[name].iloc[self.time]
            features.append(x['log_ret'])
            prices[name] = x['close']

        done = True
        if self.time < self.T: done = False
        self.time+=1
        return done, prices, np.array(features)

class FTXMarket(Market):
    pass