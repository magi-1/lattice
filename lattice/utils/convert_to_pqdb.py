import lattice.paths as paths
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
import numpy as np
import pandas as pd
import time
import pyarrow.dataset as ds
import pyarrow.compute as pc

# test having dictionary of polar df vs a concatted polars df

import polars as pl
# https://stackoverflow.com/questions/64578761/is-there-a-more-idiomatic-way-to-select-rows-from-a-pyarrow-table-based-on-conte

data_path = Path(paths.top)/'data'/'historical'/'365_days_60_seconds'
#dataset = ds.dataset(data_path, format="parquet")

data = {
    path.name.split('.')[0]: pl.read_parquet(path) for path in data_path.iterdir()
}

#cols = sorted(data['BTC_USD'].columns)

#data_list = [pl.from_pandas(_df[cols]) for _df in data.values()]


#df = pl.concat(data_list)
"""data = {
    path.name.split('.')[0]: pd.read_parquet(path) for path in data_path.iterdir()
}"""
assets = [x for x in data.keys()] # pl.Series('assets', )
#times = data['BTC_USD'].get_column('time').unique()
times = sorted(data['BTC_USD'].time[:1000])
start = time.time()
"""print(start)
for t in times:
    test = df.filter(
        (pl.col('time')==t) & 
        (pl.col('market').is_in(assets))
    )"""
for t in times:
    for asset in assets:
        test = data[asset].filter(pl.col('time')==t)
        

print(time.time()-start)



"""start = time.time()
#print(help(dataset))
table = dataset.to_table()

time_steps = pc.unique(table.column('time'))
market_table = table.filter(
    pc.and_(
        pc.equal(table.column('market'), 'BTC/USD'),
        pc.equal(table.column('market'), 'ETH/USD')
    )
)

for t in time_steps:
    assets  = market_table.filter(
        pc.equal(market_table.column('time'), pa.scalar(t))
        )
print(time.time-start)"""

# iterating through time for a given ticker
"""row_mask = pc.equal(table.column('market'), 'BTC/USD')
start = time.time()
for i in range(asset_table.num_rows):
    values = asset_table.take([i])
print(time.time)"""
    