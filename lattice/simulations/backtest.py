from lattice.broker import LocalBroker
from lattice.market import LocalMarket
from lattice.wallet import LocalWallet
from lattice.config import read_config
from lattice.investor import Investor, get_investor
from lattice.utils import logging
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
    logging.save_results(investor, name=batch_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="Name of experiment",
    )
    parser.add_argument("sims", type=int, help="Number of simulatons")
    parser.add_argument(
        "-c", "--cores", type=int, default=CORES, help="Number of cores"
    )
    args = vars(parser.parse_args())
    config = read_config(args["config"])
    wallet = LocalWallet(config["wallet"])
    market = LocalMarket(config["market"])
    broker = LocalBroker(config["broker"])
    investor = get_investor(wallet, market, broker, config["investor"])

    with mp.Pool() as pool:
        results = []
        for i in range(args["sims"]):
            results.append(
                pool.apply_async(
                    run_backtest,
                    args=(
                        investor,
                        i,
                    ),
                )
            )
        pool.close()
        pool.join()

    for r in results:
        r.get()
