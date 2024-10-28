# Strategies_combined_Hao_0409
from datamodel import OrderDepth, UserId, TradingState, Order, Position
import string
import numpy as np
import jsonpickle

class Trader:
    prev_price = []
    linear_regression_ref_num = 6
    
    def __init__(self) -> None:
        # position limit
        self.position_limit = {"STARFRUIT": 20, "AMETHYSTS": 20}

        # hardcoded linear regression coef
        # 4 parameters by curve fitting
        # self.theta = np.array([0.36532596, 0.26459464, 0.19815557, 0.17146973])
        # self.regression_constant = 2.28773387

        # 6 parameters by curve fitting
        self.regression_constant = 1.84799389
        self.theta = np.array([0.33489936, 0.23044459, 0.15290498, 0.10801622, 0.0939611, 0.07940696])
        
        # 8 parameters by curve fitting
        # self.theta = np.array([0.33041471, 0.22503081, 0.14651219, 0.09924009, 0.08077724, 0.06011376, 0.04224382, 0.01532349])
        # self.regression_constant = 1.73266517
        self.last_prices = {"STARFRUIT": [], "AMETHYSTS": []}  # initialise previous prices

    def predict_future_price(self):
        if len(self.prev_price) < self.linear_regression_ref_num:
            '''I modify here'''
            return "None"  # Not enough data
        future_price = self.regression_constant
        for i in range(len(self.theta)):
            future_price += self.theta[i] * self.prev_price[i]
        return future_price

    def run(self, state: TradingState):
        
        '''
        can the following codes be put into algorithm_linear_regression?
        We can play around with the history price, there's still many space
        '''
        
        # state.traderData restore
        if state.traderData:
            try:
                previous_state = jsonpickle.decode(state.traderData)
                self.last_prices = previous_state.get('last_prices', {})
                # print("DECODING: ", self.last_prices)
            except Exception as e:
                print(f"Error decoding traderData: {e}")

        for product, trades in state.market_trades.items():
            if trades:
                last_trade_price = trades[-1].price
                self.last_prices[product].append(last_trade_price)
                if len(self.last_prices[product]) > 100:
                    self.last_prices[product] = self.last_prices[product][-100:]

        # print("traderData: " + state.traderData)
        # print(f"traderData: {self.predict_future_price()}" + state.traderData)
        # print("Observations: " + str(state.observations))
        
        result = {}
        # orders: List[Order] = []
        traderData = ""

        starfruit_orders, _ = self.algorithm_linear_regression("STARFRUIT", state, 0.1)
        amethysts_orders, _ = self.algorithm_mean_reversion("AMETHYSTS", state, 10000, 0.1)

        result["STARFRUIT"] = starfruit_orders
        result["AMETHYSTS"] = amethysts_orders
        conversions = 1

        # encode
        traderData = jsonpickle.encode({'last_prices': self.last_prices})
        return result, conversions, traderData

    def get_buy_order_depth(self, state: TradingState, symbol: str) -> dict:
        symbol_order_depth = state.order_depths.get(symbol)
        # Check if the product exists in the order depths and get orderpaths
        if symbol_order_depth:
            symbol_buy_order_depth = symbol_order_depth.buy_orders
            return symbol_buy_order_depth
        else: return {}

    def get_sell_order_depth(self, state: TradingState, symbol: str) -> dict:
        symbol_order_depth = state.order_depths.get(symbol)
        # Check if the product exists in the order depths and get orderpaths
        if symbol_order_depth:
            symbol_sell_order_depth = symbol_order_depth.sell_orders
            return symbol_sell_order_depth
        else: return {}

    def get_median_price(self, symbol: str, state: TradingState) -> int:
        # Check if the product exists in the order depths and get orderpaths
        symbol_buy_order_depth = self.get_buy_order_depth(state, symbol)
        symbol_sell_order_depth = self.get_sell_order_depth(state, symbol)

        # Get the median price
        symbol_best_bid, symbol_bid_amount = list(symbol_buy_order_depth.items())[0]
        symbol_best_ask, symbol_ask_amount = list(symbol_sell_order_depth.items())[0]
        
        if ((symbol_bid_amount == 0) or (symbol_ask_amount == 0)):
            return 0

        # return int((symbol_best_ask * -symbol_ask_amount + symbol_best_bid * symbol_bid_amount)/ (symbol_bid_amount + -symbol_ask_amount))
        return int((symbol_best_ask + symbol_best_bid) / 2)



    '''Working!'''
    def algorithm_linear_regression(self, target_symbol: str, state: TradingState, gap: int):
        self.prev_price.append(self.get_median_price(target_symbol, state))
        if len(self.prev_price) > self.linear_regression_ref_num:
            self.prev_price.pop(0)
        
        orders: List[Order] = []
        traderData = ""

        # predict
        future_price_target_symbol = self.predict_future_price()
        if future_price_target_symbol == 'None':
            return orders, traderData
        return self.algorithm_mean_reversion(target_symbol, state, future_price_target_symbol, gap)

    '''
    Main algorithm for mean reversion
    NOTE: Highly aggressive, High frequent
    It would glance through all the market order at each iteration, 
    and buy if the order price is lower or equal to (mean - gap)
    or sell if the order price is higher or equal to (mean + gap).
    It would stop trading when there are no any trading opportunities in the market
    The current position is strictly controlled by this function
    '''
    def algorithm_mean_reversion(self, symbol: str, state: TradingState, mean: int, gap: int):
        orders: List[Order] = []
        trader_data = ""

        # get current position
        current_position = state.position.get(symbol,0)

        # get order depth
        symbol_buy_order_depth = sorted(self.get_buy_order_depth(state, symbol).items(), reverse=True)
        symbol_sell_order_depth = sorted(self.get_sell_order_depth(state, symbol).items())

        """
        Modification on trading logic 04_11:
        Now we know that we can not exceed the position limit even if we have made
        an opposition position at the same iterations
        """
        # initialize highest_bid/ask_price
        """play tricks with these parameters"""
        lowest_bid_price = mean - 1
        highest_ask_price = mean + 1
        
        # Get current position
        current_position = state.position.get(symbol, 0)
        # Process buying opportunities if the price is below the mean minus the adaptive gap
        for bid_price, bid_quantity in symbol_sell_order_depth:
            if ((bid_price <= mean - gap) and 
                current_position < self.position_limit[symbol]):
                # Calculate the maximum quantity we can buy without exceeding the position limit
                max_buy_quantity = min(-bid_quantity, self.position_limit[symbol] - current_position)

                orders.append(Order(symbol, bid_price, max_buy_quantity))
                current_position += max_buy_quantity  
                lowest_bid_price = min(bid_price, lowest_bid_price)
        
        """Something we can play tricks on: Market making!!!"""
        # Sample market making when we have more position room left
        if current_position < self.position_limit[symbol]:
            max_bid_quantity = self.position_limit[symbol] - current_position
            """NOTE: magic number below!"""
            optimal_bid_price = lowest_bid_price - 0.5
            orders.append(Order(symbol, optimal_bid_price, max_bid_quantity))


        # Reset the current position
        current_position = state.position.get(symbol, 0)
        # Process selling opportunities if the price is above the mean plus the adaptive gap
        for ask_price, ask_quantity in symbol_buy_order_depth:
            if ask_price >= mean + gap and current_position > -self.position_limit[symbol]:
                # Calculate the maximum quantity we can sell without exceeding the position limit
                    max_ask_quantity = max(-ask_quantity, - current_position - self.position_limit[symbol])
                    orders.append(Order(symbol, ask_price, max_ask_quantity))
                    current_position += max_ask_quantity  # Update the current position
                    highest_ask_price = max(ask_price, highest_ask_price)

        """Something we can play tricks on: Market making!!!"""
        # Sample market making when we have more position room left
        if current_position > -self.position_limit[symbol]:
            max_ask_quantity = - current_position - self.position_limit[symbol] 
            """NOTE: magic number below!"""
            optimal_ask_price = highest_ask_price - 0.5
            orders.append(Order(symbol, optimal_ask_price, max_ask_quantity))

        return orders, trader_data