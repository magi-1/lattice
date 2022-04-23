


class AbstractOrder(ABC):
    
    def __init__(self):
        self.open_time = 0
        self.close_time = 0
        self.open_price = 0
        self.close_price = 0
        self.profit = 0

    @abstract_method
    def place(self):
        pass

    @abstract_method
    def modify(self):
        pass

    @abstract_method
    def status(self) -> dict: # vaugue
        pass

    @abstract_method
    def cancel(self, order_id: int) -> None:
        pass


class FTXOrder(AbstractOrder):
    
    def status(self):
        # https://docs.ftx.us/#get-order-status
        pass
    
    def cancel(self):
        # https://docs.ftx.us/#cancel-order
        pass

    @abstract_method
    def place(self):
        pass

    @abstract_method
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


class LocalOrder(AbstractOrder):
    
    def status(self):
        pass
    
    def cancel(self):
        pass

    def place(self):
        pass

    def modify(self):
        pass