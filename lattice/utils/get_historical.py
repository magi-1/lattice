import lattice.paths as paths
from lattice.exchanges.ftx_client import FtxClient

import os
import time
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyarrow import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


END_DATE = datetime.datetime.fromisoformat('2022-04-21')
MAX_CALLS = 1500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution", type=int, default=60, help="Window length in seconds"
        )
    parser.add_argument(
        "--days", default=365, type=int, help="Start and end time"
        )
    args = vars(parser.parse_args())

    resolution, days = args['resolution'], args['days']

    client = FtxClient(
        api_key=os.environ['FTX_DATA_KEY'], 
        api_secret=os.environ['FTX_DATA_SECRET']
    )

    print('Connected to client')
    start_date = END_DATE - datetime.timedelta(days=days)
    t0, t1 = start_date.timestamp(), END_DATE.timestamp()
    num_obs = (t1-t0)/resolution
    num_chunks = int(np.ceil(num_obs/MAX_CALLS))+1
    time_bounds = np.linspace(t0, t1, num_chunks)

    markets = pd.read_csv(paths.data/'markets.csv')
    for market in markets['name']:
        print(f'Pulling last {days} days of {market}...')
        dfs = []
        for i in tqdm(range(num_chunks-1)):
            prices = client.get_historical_prices(
                market=market, 
                resolution=resolution, 
                start_time=time_bounds[i],
                end_time=time_bounds[i+1]
            )  
            _df = pd.DataFrame(prices)
            _df['market'] = market
            dfs.append(_df)

        data = pd.concat(dfs)
        dirname = paths.data/'historical'/f'{days}_days_{resolution}_seconds'
        dirname.mkdir(parents=True, exist_ok=True)
        fname = market.replace('/','_')
        fpath = dirname/(f'{fname}.parquet')
        data.to_parquet(fpath, engine='pyarrow', index=False)
        print(f"Saved {data.shape[0]} rows\n")