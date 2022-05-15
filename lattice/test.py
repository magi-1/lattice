from lattice.market import FTXMarket
from lattice.config import read_config
from lattice.order import FTXMarketOrder
from lattice.features import feature_registry

"""order = FTXMarketOrder(
    market='BTC/USD',
    size=0.0062,
    side='buy'
)
order.place()"""


print(feature_registry)

config = read_config('my_ftx_strategy')
market = FTXMarket(config=config['market'])
g = market.get_state()

print(g)