from lattice.wallet import Wallet
import lattice.paths as paths

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import datetime
import functools
import warnings


"""
Think of lattice.markets as customizable datastreams that restrict
an agent/investor to a specific subset of the market.

Later on these can also take config['market']['features'] = List[str]
and call feature classes to pipe out the data. Can also let the market have
a buffer mechanism if desired so that recurrent data can be used for a given model.
This too can be specified by config string -> constructor -> buffer initialized inside market obj.
"""


class Market(ABC):

    """
    TODO: Impliment base market methods / properties

    - Takes as input a market config specifying either
      the whole market or a subset of the market
    - Handles the data streams / asynchronous code
    """

class LocalMarket(Market):
   
    def __init__(self, config) -> None:
        self.__dict__.update(config)
        self.data = self._load_data()
        self.T = self.data['BTC_USD'].shape[0]
        self.time = 0
        
    def to_timestamp(self, iso_time: str):
        return datetime.datetime.fromisoformat(iso_time).timestamp()*1000
        
    def _load_data(self):
        t0,t1 = list(map(self.to_timestamp, self.window))
        data_dir = paths.data/'historical'/self.dataset
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

        """
        Potentially have this read from sqlite db instead?
        Having the dataframes in a dictionary is pretty slow.
        """
        prices, features = dict(), []
        time = self.data[self.assets[0]].iloc[self.time]['time']
        for name in self.assets:
            x = self.data[name].iloc[self.time]
            features.append(x['log_ret'])
            prices[name] = x['close']

        done = False
        if self.time >= self.T-1: 
            done = True
        self.time += 1
        return done, time, prices, np.array(features)


class FTXMarket(Market):
    # async code which will cause the investor polling mechanism to await
    pass