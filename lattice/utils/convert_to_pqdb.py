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

# https://stackoverflow.com/questions/64578761/is-there-a-more-idiomatic-way-to-select-rows-from-a-pyarrow-table-based-on-conte

data_path = Path(paths.top)/'data'/'365_days_60_seconds'
dataset = ds.dataset(data_path, format="parquet")

start = time.time()
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
print(time.time-start)

# iterating through time for a given ticker
"""row_mask = pc.equal(table.column('market'), 'BTC/USD')
start = time.time()
for i in range(asset_table.num_rows):
    values = asset_table.take([i])
print(time.time)"""
    