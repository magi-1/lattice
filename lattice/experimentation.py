from lattice.market import LocalMarket
from lattice.config import read_config
from lattice.order import FTXMarketOrder
from lattice.features import feature_registry
from lattice.models import gnn


config = read_config("local_gnn")
market = LocalMarket(config=config["market"])
_, _, _, features = market.get_state()
graph = gnn.construct_graph(features)
print(graph)
