from lattice.clients import FtxClient
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import uuid
import os
load_dotenv()


class Order(ABC):
    
    def __init__(
        self,
        market: str,
        side: str,
        size: float,
        open_price: float,
        fee: float=0.0
    ):  
        self.id = uuid.uuid4()
        self.market = market
        self.side = side
        self.size = size
        self.open_price = open_price
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
        return self.market.split('_')

    def market_delta(self, current_price: float):
        return self.sign*(current_price-self.open_price)


class LocalMarketOrder(Order):

    def __init__(
        self,
        market: str,
        side: str,
        size: float,
        open_price: float,
        open_time: float,
        fee: float
    ):
        super().__init__(market, side, size, open_price, fee)
        self.open_time = open_time
    
    def place(self):
        return {'success':True}
    
    def cancel(self):
        return {'success':True}


class LocalTriggerOrder(Order):

    def place(self):
        return {'success':True}
    
    def cancel(self):
        return {'success':True}


class FTXOrder(Order):
    
    client = FtxClient(
        api_key=os.environ['FTX_TRADE_KEY'], 
        api_secret=os.environ['FTX_TRADE_SECRET']
    )

    def __init__(
        self,
        market: str,
        side: str,
        size: float,
        open_price: float,
    ):
        super().__init__(market, side, size, open_price)

    def status(self):
        return self.client.get_order_status(self.id)
    
    def cancel(self):
        return self.client.cancel_order(self.id)

    @abstractmethod
    def place(self):
        pass


class FTXMarketOrder(FTXOrder):

    def __init__(
        self,
        market: str,
        side: str,
        size: float,
    ):
        super().__init__(market, side, size, open_price=0)

    def place(self):

        self.client.place_order(
            market=self.market,
            side=self.side,
            size=self.size,
            type='market',
            price=None
        )


class FTXLimitOrder(FTXOrder):

    def __init__(
        self,
        market: str,
        side: str,
        size: float,
        open_price: float,
    ):
        super().__init__(market, side, size, open_price)

    def place(self):
        self.client.place_order(
            market=self.market,
            side=self.side,
            price=self.open_price,
            size=self.size,
            type='limit'
        )


class FTXTriggerOrder(FTXOrder):
    
  
    def cancel(self):
        # https://docs.ftx.us/#cancel-order
        pass

    def place(self):
        # https://docs.ftx.us/#place-trigger-order
        pass

    def modify(self):
        # https://docs.ftx.us/#modify-trigger-order
        pass  
