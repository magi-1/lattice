from abc import ABC, abstract_method

"""
All markets take the same kind of config then know
how to interact with the real/synthetic market by pulling the
data down at a given rate or only piping out a subset of tickers. 

Later on these can also take config['market']['features'] = List[str]
and call feature classes to pipe out the data. Can also let the market have
a buffer mechanism if desired so that recurrent data can be used for a given model.

^ This also falls inline with having methods to extract and process orderbook information.
"""

class AbstractMarket(ABC):

    """
    - Takes as input a market config specifying either
      the whole market or a subset of the market
    - Handles the data streams / asynchronous code
    """

    def __init__(self, config):
        self.config = config


class LocalMarket(AbstractMarket):
    pass

class FTXMarket(AbstractMarket):
    pass