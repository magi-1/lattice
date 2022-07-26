from lattice.broker import LocalBroker
from lattice.market import LocalMarket
from lattice.wallet import LocalWallet
from lattice.config import read_config
from lattice.investor import Investor, get_investor
from lattice.utils import log
import lattice.paths as paths

import os
import jax
import jax.numpy as jnp
import time
import jraph
import optax
import argparse
import numpy as np
import haiku as hk
import multiprocessing as mp
from typing import List, Union


CORES = int(mp.cpu_count())


def run_episode(investor: Investor, batch_id=None) -> list:
    np.random.seed(os.getpid())
    while experience := investor.evaluate_market():
        if not type(experience) == bool:
            break
    # log.save_results(investor, name=batch_id)
    return experience


def run_episodes(investor: Investor) -> List[log.ExperienceBuffer]:
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
def loss(params, graphs, actions, rewards):
    logits = network.apply(params, graphs)
    log_prob = jax.nn.log_softmax(logits)
    prods = jnp.take_along_axis(log_prob, jnp.expand_dims(actions, axis=1), axis=1)*rewards
    return -jnp.mean(prods)

@jax.jit
def update(params, opt_state, graphs, actions, rewards):
    grads = jax.grad(loss)(params, graphs, actions, rewards)
    updates, opt_state = opt_update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state


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
        if i == 0:
            # Initializing investor / network
            _ = run_episode(investor, i)
            network, params = investor.network, investor.params 
            opt_init, opt_update = optax.adam(2e-4)
            opt_state = opt_init(params)
            investor.reset_env()
        else:
            graphs, actions, rewards = run_episode(investor, i)
            params, opt_state = update(params, opt_state, graphs, actions, rewards)
            investor.reset()
            investor.params = params # FIXME: Gross

            if i%100==0:
                investor.save_params()
