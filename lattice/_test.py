from lattice.market import FTXMarket
from lattice.config import read_config
from lattice.order import FTXMarketOrder


"""order = FTXMarketOrder(
    market='BTC/USD',
    size=0.0001,
    side='buy'
)
order.place()"""

config = read_config('ftx_bot')
market = FTXMarket(config=config['market'])
data = market.get_data('BTC/USD')
historical_data = market.get_window('BTC/USD')

print(historical_data)