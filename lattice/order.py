from abc import ABC, abstractmethod
import uuid


class Order(ABC):
    
    def __init__(
        self,
        asset: str,
        side: str,
        size: float,
        open_price: float,
        otype: str,
        fee: float
    ):  
        self.id = uuid.uuid4()
        self.asset = asset
        self.side = side
        self.size = size
        self.open_price = open_price
        self.otype = otype
        self.fee = fee
        self.open_time = None
        self.close_time = None
        self.close_price = None
        self.profit = None

    """
    def stamp(self, time): Can do this later for ftx probably 
    local time needs to be passed through but ftx doesnt, time.time()
    """

    @abstractmethod
    def cancel(self, order_id: int) -> None:
        pass

    @abstractmethod
    def place(self):
        pass

    @abstractmethod
    def modify(self):
        pass

    @property
    def sign(self):
        return 1 if self.side == 'BUY' else -1

    @property
    def value(self):
        return self.sign*self.open_price*self.size
    
    @property
    def amount(self):
        return self.sign*self.size
    
    def components(self):
        return self.asset.split('_')

    def market_delta(self, current_price: float):
        return self.sign*(current_price-self.open_price)


class LocalOrder(Order):

    def __init__(
        self,
        asset: str,
        side: str,
        size: float,
        open_price: float,
        open_time: float,
        otype: str,
        fee: float
    ):
        super().__init__(asset, side, size, open_price, otype, fee)
        self.open_time = open_time
    
    def place(self):
        response = {'success':True}
        return response
    
    def cancel(self):
        response = {'success':True}
        return response

    def modify(self):
        # TODO
        pass


"""
class LocalTriggerOrder(Order):

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
"""