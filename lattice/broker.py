from lattice.utils.conversions import to_timestamp
from lattice.config import BrokerConfig
from lattice.order import *

from abc import ABC, abstractmethod


class Broker(ABC):


    """
    The broker is the mediator and enforces trading rate limits as well 
    as places the trades. Records the orders as well. 
    """

    def __init__(self, config: BrokerConfig):
        self.__dict__.update(config)
        self.orders = dict()

    def place_order(self, order: Order):
        if order:
            response = order.place()
            if response['success']:
                self.orders.setdefault(order.id, order)
            else:
                print('Failed to place order!')
            
    def cancel_order(self, order_id: str):
        response = order.cancel()
        if response['success']:
            del self.orders[order_id]
        else:
            print('Failed to cancel order!')

    @abstractmethod
    def market_order(self):
        pass


class LocalBroker(Broker):

    def __init__(self, config):
        super().__init__(config)
        
    def market_order(
        self, 
        market: str, 
        side: str,
        size: float,
        open_price: float,
        open_time: float,
    ) -> LocalMarketOrder:  
        return LocalMarketOrder(
            market=market, 
            side=side, 
            size=size,
            open_price=open_price,
            open_time=open_time,
            fee=self.fee
            )


class FTXBroker(Broker):

    def __init__(self, config):
        super().__init__(config)

    def place_order(self, order: Order) -> bool:
        if isinstance(order, FTXMarketOrder):
            response = order.place()
            if response['success']:
                result = response['result']
                order.id = result['id']
                order.open_price = result['price']
                order.open_time = to_timestamp(result['created_at'])
                self.orders.setdefault(order.id, order)
                return True
            else:
                print('Failed to place order!')
                return False
        
    def market_order(
        self, 
        market: str, 
        side: str,
        size: float
    ) -> LocalMarketOrder:  
        return FTXMarketOrder(
            market=market, 
            side=side, 
            size=size,
            )

    def limit_order(
        self, 
        market: str, 
        side: str,
        size: float,
        open_price: float
    ) -> LocalMarketOrder:  
        return FTXLimitOrder(
            market=market, 
            side=side, 
            size=size,
            open_price=open_price
            )