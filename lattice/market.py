from lattice.utils.conversions import to_timestamp
from lattice.config import MarketConfig
from lattice.features import feature_registry
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


class Market(ABC):

    def __init__(self, config: MarketConfig) -> None:
        self.__dict__.update(config)
        self.feature_set = self.init_features()

    def init_features(self) -> None:
        return [feature_registry[name](params) for name, params in self.features.items()]
        
    def compute_features(self, prices: pl.Series, volumes: pl.Series) -> np.ndarray:
        features = [f.evaluate(prices, volumes) for f in self.feature_set]
        return features

    def pivot(self, df: pl.DataFrame, column: str) -> np.ndarray:
        return df.pivot(
            values=column, index='startTime', columns='market'
            ).drop('startTime').to_numpy()

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
        prices = self.pivot(big_df, 'close')
        volumes = self.pivot(big_df, 'volume')
        features = self.compute_features(prices, volumes)
        self.markets = prices.columns # might be redundant
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

    def get_market_data(self, market: str) -> pl.DataFrame:
        curr_time = int(time.time())
        prices = self.client.get_historical_prices(
            market=market,
            resolution=self.resolution,
            start_time=int(curr_time-self.dt),
            end_time=curr_time
        )
        return pl.DataFrame(prices)

    def get_orderbook_data(self, market: str):
        if self.ob_depth:
            data = self.client.get_orderbook(
                market=market,
                depth=self.ob_depth
            )   
            columns = ['bid_price','bid_volume','ask_price', 'ask_volume']
            values = np.concatenate((data['bids'], data['asks']), axis=1)
            return pl.DataFrame(values, columns=columns)
        
    def get_state(self):
        mkt_dfs = []
        #ob_dfs = []
        for market in self.markets:
            mkt_df = self.get_market_data(market)
            #ob_df = self.get_orderbook_data(market)
            market_col = pl.lit(market).alias("market")
            mkt_df = mkt_df.with_column(market_col)
            #ob_df = ob_df.with_column(market_col)
            mkt_dfs.append(mkt_df)
            #ob_dfs.append(ob_df)

        big_mkt_df = pl.concat(mkt_dfs)
        #big_ob_df = pl.concat(ob_dfs)
        
        prices = self.pivot(big_mkt_df, 'close')
        volumes = self.pivot(big_mkt_df, 'volume')
        features = self.compute_features(prices, volumes)
        return prices, features
