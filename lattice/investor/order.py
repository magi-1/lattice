from abc import ABC, abstractmethod


class Order(ABC):
    
    def __init__(
        self,
        asset: str,
        side: str,
        size: float,
        open_price: float,
        open_time: float
    ):
        self.asset = asset
        self.side = side
        self.size = size
        self.open_price = open_price
        self.open_time = open_time
        self.close_time = None
        self.close_price = None
        self.profit = None

    
    def modify(self):
        """ To do later on """

    def status(self):
        """ To do later on """

    def cancel(self, order_id: int) -> None:
        """ To do later on """
    
    @abstractmethod
    def place(self):
        pass

    @property
    def sign(self):
        return 1 if self.side == 'BUY' else -1

    @property
    def value(self):
        return self.sign*self.open_price*self.size

    def profit(self, current_price: float):
        return self.sign*(current_price-self.open_price)


class LocalOrder(Order):

    def place(self):
        pass


class FTXOrder(Order):
    
    def status(self):
        # https://docs.ftx.us/#get-order-status
        pass
    
    def cancel(self):
        # https://docs.ftx.us/#cancel-order
        pass

    @abstractmethod
    def place(self):
        pass

    @abstractmethod
    def modify(self):
        pass


class FTXMarketOrder(FTXOrder):

    def place(self):
        #https://docs.ftx.us/#place-order
        pass

    def modify(self):
        # https://docs.ftx.us/#modify-order
        pass


class FTXTriggerOrder(FTXOrder):
    
    def status(self):
        # https://docs.ftx.us/#get-order-status
        pass
    
    def cancel(self):
        # https://docs.ftx.us/#cancel-order
        pass

    def place(self):
        # https://docs.ftx.us/#place-trigger-order
        pass

    def modify(self):
        # https://docs.ftx.us/#modify-trigger-order
        pass