from lattice.utils.conversions import to_timestamp
from lattice.config import MarketConfig
from lattice.clients import FtxClient
from lattice.wallet import Wallet
import lattice.paths as paths

from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Dict
import polars as pl
import numpy as np
import functools
import datetime
import warnings
import time
import os
load_dotenv()


"""
Think of lattice.markets as customizable datastreams that restrict
an agent/investor to a specific subset of the market.

Later on these can also take config['market']['features'] = List[str]
and call feature classes to pipe out the data. Can also let the market have
a buffer mechanism if desired so that recurrent data can be used for a given model.
This too can be specified by config string -> constructor -> buffer initialized inside market obj.
"""


class Market(ABC):

    def __init__(self, config: MarketConfig) -> None:
        self.__dict__.update(config)

    def compute_features(self, data: pl.DataFrame) -> np.ndarray:
        return data

    @abstractmethod
    def get_state(self):
        pass


class LocalMarket(Market):
   
    def __init__(self, config) -> None:
        super().__init__(config)
        self.prices, self.features = self.load_data()

    def load_data(self):
        data_dir = paths.data/'historical'/self.dataset
        t0, t1 = list(map(self.to_timestamp, self.window)) 
        condtion = (pl.col('time') >= t0) & (pl.col('time') < t1)

        dataframes = []
        for path in data_dir.iterdir():
            market_name = path.stem
            if market_name in self.markets:
                df = pl.read_parquet(path).filter(condtion)
                df = df.with_column(pl.lit(market_name).alias("market"))
                dataframes.append(df)

                if init_bool:=True:
                    self.timesteps = df.get_column('time').unique().sort().to_list()
                    self.t, self.T = 0, len(self.timesteps) 
                    init_bool = False

        big_df = pl.concat(dataframes)
        prices = big_df.pivot(
            values='close', index='startTime', columns='market'
            ).drop('startTime')
        self.markets = prices.columns # might be redundant
        features = self.compute_features(big_df)
        return prices.to_numpy(), features

    def get_state(self):
        time = self.t
        current_prices = dict(zip(self.markets, self.prices[self.t]))
        features = [] # self.featues[self.t]
        
        done = False
        if self.t >= self.T-1: 
            done = True

        self.t += 1
        return done, time, current_prices, features


class FTXMarket(Market):
    
    client = FtxClient(
        api_key=os.environ['FTX_DATA_KEY'], 
        api_secret=os.environ['FTX_DATA_SECRET']
    )

    def __init__(self, config) -> None:
        super().__init__(config)
        self.dt = datetime.timedelta(hours=self.window_size).total_seconds()

    def get_data(self, market: str):
        raw_data = self.client.get_market(market)
        return raw_data

    def get_window(self, market: str) -> pl.DataFrame:
        curr_time = int(time.time())
        prices = self.client.get_historical_prices(
            market=market,
            resolution=self.resolution,
            start_time=int(curr_time-self.dt),
            end_time=curr_time
        )
        return pl.DataFrame(prices)

    def get_state(self):
        dataframes = []
        for market in self.markets:
            lagged_data = self.get_window(market)
            lagged_data['market'] = market
            dataframes.append(lagged_data)
        big_df = pl.concat(dataframes)
        prices = big_df.pivot(
            values='close', index='startTime', columns='market'
            ).drop('startTime')
        features = self.compute_features(big_df)
        return prices, features