from typing import List, NewType,
from abc import ABC, abstract_method


class AbstractWallet(ABC):

    def __init__(self, capital: float) -> None:
        self.capital = capital

    @abstract_method
    def process_config(self):
        # Converts the config into the same data structue defined by
        # the abstract wallet so that methods can be shared.
        pass

    def calculate_something(self):
        pass


class FTXWallet(AbstractWallet):
    
    """
    Gets initialized with FTX wallet data.
    """

    def __init__(self):
        pass

    def process_config(self):
        pass


class LocalWallet(AbstractWallet):
    
    """
    This config will allow you to partition your account
    via proportions in [0,1] that divy up the total_assets.
    """

    def __init__(self, config):
        self.config = config

    def process_config(self):
        pass




