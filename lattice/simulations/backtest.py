
from lattice.broker import LocalBroker
from lattice.market import LocalMarket
from lattice.wallet import LocalWallet
import lattice.utils.plotting as plot
from lattice.config import read_config
from lattice.investor import Investor, get_investor
import lattice.paths as paths

import multiprocessing as mp
import numpy as np
import argparse
import time
import os


CORES = int(mp.cpu_count())


def run_backtest(investor: Investor, batch_id=None) -> None:
    """
    Takes in a configured investor and executes trading scenario. 
    """
    np.random.seed(os.getpid())

    # Executing backtest
    while total_value := investor.evaluate_market():
        pass

    # Logging data
    log_results(investor, batch_id=batch_id)


def log_results(investor: Investor, batch_id=None):
    # arg: run_id=None maybe add later if needed for RL
    """
    Standardize and build upon this
    """ 

    # Setting out directory
    save_path = paths.data/'sim_out'
    if batch_id != None:
        save_path /= f'sim_{batch_id}'

    try:
        save_path.mkdir(parents=True, exist_ok=False)
    except:
        msg = "Sim data already exists. Try running 'make clear_sims'"
        raise OSError(msg)

    # Writing data
    history = investor.wallet.get_history()
    history.to_parquet(save_path/'wallet_history.parquet', index=False)

    # Saving visualizations
    plot.visualize_backtest(history, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Name of experiment",
        )
    parser.add_argument(
        "-n", "--num_sims", type=int, help="Number of simulatons"
    )
    parser.add_argument(
        "-c", "--cores", type=int, default=CORES, help="Number of cores"
    )
    args = vars(parser.parse_args())

    config = read_config(args['config'])
    wallet = LocalWallet(config['wallet'])
    market = LocalMarket(config['market'])
    broker = LocalBroker(config['broker'])
    investor = get_investor(wallet, market, broker, config['investor'])
    
    with mp.Pool() as pool:
        results = []
        for i in range(args['num_sims']):
            results.append(
                pool.apply_async(run_backtest, args=(investor, i))
            )
        pool.close()
        pool.join()

    for r in results:
        r.get()
