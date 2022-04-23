from abc import ABC, abstract_method


class Investor(ABC):

    """
    Takes a (wallet, market) and places trades.
    All of the input objects have generic methods that generalize
    accross their class types such that investors can be written generally
    for any exchange whether it be local or online.
    """

    def __init__(self, wallet: Wallet=None, market: Market, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.wallet = wallet
        self.orders = Orders()

    def open(self, order: Order):
        # Operates on abstract orders
        pass

    def close(self, order: Order):
        # Operates on abstract orders
        pass

    @abstract_method
    def evaluate_market(self):
        # Operates on abstract markets but is customizable
        pass