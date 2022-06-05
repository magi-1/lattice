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


def run_episode(investor: Investor, batch_id=None) -> list:
    np.random.seed(os.getpid())
    while experience := investor.evaluate_market():
        pass
    # logging.save_results(investor, name=batch_id)
    return experience


def run_episodes(investor: Investor) -> List[logging.ExperienceBuffer]:
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
    return results


@jax.jit
def loss(params, graph, reward):
    logits = network.apply(params, graph)
    log_prob = jax.nn.log_softmax(logits) * reward
    return -jnp.sum(log_prob) / len(log_prob)


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

    for i in range(args["sims"]):
        experience = run_episode(investor, i)
        if i == 0:
            opt_init, opt_update = optax.adam(2e-4)
            opt_state = opt_init(investor.params)
        else:
            experience = run_episode(investor, i)
