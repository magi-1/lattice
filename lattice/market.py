from lattice.utils.conversions import to_timestamp
from lattice.config import MarketConfig
from lattice.features import feature_registry
from lattice.clients import FtxClient
from lattice.wallet import Wallet
import lattice.paths as paths

from abc import ABC, abstractmethod
from collections import OrderedDict
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

    def init_features(self) -> OrderedDict:
        return OrderedDict(
            {
                name: feature_registry[name](params)
                for name, params in self.features.items()
            }
        )

    def compute_features(self, prices: np.ndarray, volumes: np.ndarray) -> OrderedDict:
        return OrderedDict(
            {f: f.evaluate(prices, volumes) for name, f in self.feature_set.items()}
        )

    def pivot(self, df: pl.DataFrame, column: str) -> np.ndarray:
        return (
            df.pivot(values=column, index="startTime", columns="market")
            .drop("startTime")
            .to_numpy()
        )

    @property
    def num_markets(self):
        return len(self.markets)
        
    @abstractmethod
    def get_state(self):
        pass


class LocalMarket(Market):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.prices, self.volumes = self.load_data()

    def reset(self):
        self.t = self.lag

    def load_data(self):
        data_dir = paths.data / "historical" / self.dataset
        t0, t1 = list(map(to_timestamp, self.window))
        condtion = (pl.col("time") >= t0) & (pl.col("time") < t1)

        mkt_dfs = []
        for path in data_dir.iterdir():
            market_name = path.stem
            if market_name in self.markets:
                df = pl.read_parquet(path).filter(condtion)
                df = df.with_column(pl.lit(market_name).alias("market"))
                mkt_dfs.append(df)

                if init_bool := True:
                    self.timesteps = df.get_column("time").unique().sort().to_list()
                    self.t, self.T = self.lag, len(self.timesteps)
                    init_bool = False

        big_mkt_df = pl.concat(mkt_dfs)
        prices = self.pivot(big_mkt_df, "close")
        volumes = self.pivot(big_mkt_df, "volume")
        return prices, volumes

    def get_state(self):
        current_prices = dict(zip(self.markets, self.prices[self.t]))
        start_index, end_index = self.t - self.lag + 1, self.t + 1

        price_window = self.prices[start_index:end_index]
        volume_window = self.volumes[start_index:end_index]
        features = self.compute_features(price_window, volume_window)

        # Simulation state
        done = False
        if self.t >= self.T - 2:
            done = True
        self.t += 1
        return done, self.t, current_prices, features


class FTXMarket(Market):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.client = FtxClient(
            api_key=os.environ["FTX_DATA_KEY"], api_secret=os.environ["FTX_DATA_SECRET"]
        )
        self.dt = datetime.timedelta(seconds=self.lag * self.resolution).total_seconds()

    def get_lagged_market_data(self, market: str) -> pl.DataFrame:
        curr_time = int(time.time())
        prices = self.client.get_historical_prices(
            market=market,
            resolution=self.resolution,
            start_time=int(curr_time - self.dt),
            end_time=curr_time,
        )
        return pl.DataFrame(prices)

    def get_orderbook_data(self, market: str):
        if self.ob_depth:
            data = self.client.get_orderbook(market=market, depth=self.ob_depth)
            columns = ["bid_price", "bid_volume", "ask_price", "ask_volume"]
            values = np.concatenate((data["bids"], data["asks"]), axis=1)
            return pl.DataFrame(values, columns=columns)

    def get_state(self):
        mkt_dfs, ob_dfs = [], []
        for market in self.markets:
            mkt_df = self.get_lagged_market_data(market)
            market_col = pl.lit(market).alias("market")
            mkt_df = mkt_df.with_column(market_col)
            mkt_dfs.append(mkt_df)

        big_mkt_df = pl.concat(mkt_dfs)
        prices = self.pivot(big_mkt_df, "close")
        volumes = self.pivot(big_mkt_df, "volume")
        features = self.compute_features(prices, volumes)

        done = False
        current_time = int(time.time())
        curent_prices = dict(zip(self.markets, prices[-1]))
        return done, current_time, curent_prices, features
