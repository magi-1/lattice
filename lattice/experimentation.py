from lattice.market import LocalMarket
from lattice.config import read_config
from lattice.order import FTXMarketOrder
from lattice.features import feature_registry
from lattice.models import gnn

import jax
import haiku as hk
import numpy as np

config = read_config("local_gnn")
market = LocalMarket(config=config["market"])
_, _, _, features = market.get_state()

graph = gnn.construct_graph(features, jax.numpy.array([[0.0, 0.0]]))
network = hk.without_apply_rng(hk.transform(gnn.network_definition))
params = network.init(jax.random.PRNGKey(42), graph)
out = network.apply(params, graph)

print(out)
