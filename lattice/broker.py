from abc import ABC, abstractmethod
from lattice.order import *


class Broker(ABC):

    def __init__(self, config: dict):
        self.config = config
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
    def create_order(self):
        pass


class LocalBroker(Broker):

    def __init__(self, config: dict):
        super().__init__(config)
        
    def create_order(
        self, 
        asset: str, 
        side: str,
        size: float,
        open_price: float,
        open_time: float,
        otype: str = 'market'
    ) -> LocalOrder:  
        return LocalOrder(
            asset=asset, 
            side=side, 
            size=size,
            open_price=open_price,
            open_time=open_time,
            otype=otype
            )