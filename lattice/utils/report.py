import lattice.utils.plotting as plot
import lattice.paths as paths
import polars as pl


DATA_PATH = paths.data
REPORT_PATH = DATA_PATH / "report"
REPORT_PATH.mkdir(exist_ok=True)

if __name__ == "__main__":

    history_files = (DATA_PATH / "sim_out").rglob("**/wallet_history.parquet")

    returns = []
    for path in history_files:
        df = pl.read_parquet(path)
        wallet_values = df.get_column("total_value")
        start_val, end_val = wallet_values.take([0, df.height - 1])
        ret = (end_val - start_val) / start_val
        returns.append(ret)

    plot.plot_return_distribution(returns, REPORT_PATH)
