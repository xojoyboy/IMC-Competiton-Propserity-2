# Strategies_combined_Hao_0409
from datamodel import OrderDepth, UserId, TradingState, Order, Position
import string
import numpy as np
import jsonpickle
import pandas as pd
from typing import List, Dict

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
        self.last_prices: Dict[str, List[int]] = {"STARFRUIT": [], "AMETHYSTS": []}
        self.atr_multiplier = 0.5  # initialise previous prices

        self.market_analysis = {
            "AMETHYSTS": {
                'SMA': 9999.7,
                'EMA': 9999.498935442854,
                'RSI': 45.945945945945944,
                'Bollinger_Upper': 10001.9208400562,
                'Bollinger_Lower': 9997.979159943801,
                'Volatility': 0.985420028099787,
                'Predicted_Mid_Price': 9999.68102296506,
                'Regression_Coefficients': np.array([1.01132775e+04, 7.50277671e-02, -1.16870591e-02, 4.36132880e-03, -7.90335893e-02])
            },
            "STARFRUIT": {
                'SMA': 5043.9,
                'EMA': 5044.148286492216,
                'RSI': 55.0,
                'Bollinger_Upper': 5046.2824559216415,
                'Bollinger_Lower': 5042.617544078358,
                'Volatility': 0.9162279608207858,
                'Predicted_Mid_Price': 5052.8361918519295,
                'Regression_Coefficients': np.array([51.51121501, 0.87492925, -0.49278925, 0.52432386, 0.08366426])
            }
        }

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

        starfruit_orders, _ = self.algorithm_linear_regression("STARFRUIT", state)
        amethysts_orders, _ = self.algorithm_mean_reversion("AMETHYSTS", state, 10000)

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
    def algorithm_linear_regression(self, target_symbol: str, state: TradingState):
        self.prev_price.append(self.get_median_price(target_symbol, state))
        if len(self.prev_price) > self.linear_regression_ref_num:
            self.prev_price.pop(0)
        
        orders: List[Order] = []
        traderData = ""

        # predict
        future_price_target_symbol = self.predict_future_price()
        if future_price_target_symbol == 'None':
            return orders, traderData
        return self.algorithm_mean_reversion(target_symbol, state, future_price_target_symbol)

    '''
    Main algorithm for mean reversion
    NOTE: Highly aggressive, High frequent
    It would glance through all the market order at each iteration, 
    and buy if the order price is lower or equal to (mean - gap)
    or sell if the order price is higher or equal to (mean + gap).
    It would stop trading when there are no any trading opportunities in the market
    The current position is strictly controlled by this function
    '''
    def algorithm_mean_reversion(self, symbol: str, state: TradingState, mean: int):
        orders = []
        trader_data = ""

        # Dynamically calculate the gap based on ATR and additional market factors
        gap = self.get_dynamic_gap(symbol)

        # Dynamically adjust order sizes based on liquidity
        # liquidity_adjustment = self.adjust_order_size_based_on_liquidity(symbol, state)

        # Get order depth
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
    
    def get_dynamic_gap(self, symbol: str):
        """Enhances gap calculation by considering volatility, Bollinger Bands width, and ATR."""
        volatility = self.market_analysis[symbol]['Volatility']
        bollinger_width = self.market_analysis[symbol]['Bollinger_Upper'] - self.market_analysis[symbol]['Bollinger_Lower']
        atr = self.calculate_atr(symbol)
        # Adjust the multiplier based on the volatility and bollinger width to make the strategy more adaptive
        multiplier = max(0.5, min(volatility, bollinger_width / atr))
        return atr * multiplier

    def calculate_atr(self, product: str, window: int = 14) -> float:
        if len(self.last_prices[product]) < window + 1:
            return 0.01  # Default small value for insufficient data
        tr_list = [abs(self.last_prices[product][i] - self.last_prices[product][i-1]) for i in range(1, len(self.last_prices[product]))]
        return np.mean(tr_list[-window:])
    
    def calculate_liquidity(self, product: str, state: TradingState):
        order_depth = state.order_depths.get(product)
        if not order_depth: 
            return 0  # Assume no liquidity if there's no order depth

        # Sum volumes of the top N levels of the order book on both sides
        top_n_levels = 5
        buy_liquidity = sum([volume for _, volume in sorted(order_depth.buy_orders.items(), reverse=True)[:top_n_levels]])
        sell_liquidity = sum([volume for _, volume in sorted(order_depth.sell_orders.items())[:top_n_levels]])

        # Average liquidity
        avg_liquidity = (buy_liquidity + sell_liquidity) / 2
        return avg_liquidity

    def adjust_order_size_based_on_liquidity(self, product: str, state: TradingState):
        liquidity = self.calculate_liquidity(product, state)
        
        # Example thresholds based on your analysis
        low_liquidity_threshold = 10  # Adjust based on your market analysis
        high_liquidity_threshold = 50  # Adjust based on your market analysis

        if liquidity < low_liquidity_threshold:
            return 0.5  # Reduce order size to 50% in low liquidity
        elif liquidity > high_liquidity_threshold:
            return 1.5  # Increase order size by 50% in high liquidity
        else:
            return 1.0
