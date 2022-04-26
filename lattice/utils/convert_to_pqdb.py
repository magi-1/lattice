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

data_path = Path(paths.top)/'data'/'365_days_60_seconds'
dataset = ds.dataset(data_path, format="parquet")

start = time.time()
#print(help(dataset))
table = dataset.to_table().to_pandas()
print(table.columns)
# https://stackoverflow.com/questions/64578761/is-there-a-more-idiomatic-way-to-select-rows-from-a-pyarrow-table-based-on-conte
