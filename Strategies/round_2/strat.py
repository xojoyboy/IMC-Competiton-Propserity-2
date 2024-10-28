from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np
import jsonpickle
from typing import List, Dict, Any
import json
import math

# LOGGER --------------------------------------------------------------------------------------------

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

# LOGGER --------------------------------------------------------------------------------------------

PRICE_AGGRESSION = {
    "AMETHYSTS": 0,
    "STARFRUIT": 0,
    "ORCHIDS": 0
}

class Trader:
    prev_price = []
    linear_regression_ref_num = 6
    previous_starfruit_prices = []

    def __init__(self) -> None:
        # position limit
        self.position_limit = {"STARFRUIT": 20, "AMETHYSTS": 20, "ORCHIDS": 100}
        self.POSITION = {'AMETHYSTS': 0, 'STARFRUIT': 0, 'ORCHIDS': 0}

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
        self.last_prices: Dict[str, List[int]] = {"STARFRUIT": [], "AMETHYSTS": [], "ORCHIDS": []}
        self.atr_multiplier = 0.5  # initialise previous prices

        self.market_analysis = {
            "AMETHYSTS": {
                'Bollinger_Upper': 10001.9208400562,
                'Bollinger_Lower': 9997.979159943801,
                'Volatility': 0.985420028099787,
            },
            "STARFRUIT": {
                'Bollinger_Upper': 5046.2824559216415,
                'Bollinger_Lower': 5042.617544078358,
                'Volatility': 0.9162279608207858,
            },
            "ORCHIDS": {
                'Bollinger_Upper': 1054.4201205917313,
                'Bollinger_Lower': 1031.7548794082688,
                'Volatility': 5.666310295865586,
            }
        }

    def update_starfruit_price_history(self, previousTradingState, tradingState: TradingState):
        if "previous_starfruit_prices" in previousTradingState:
            self.previous_starfruit_prices = previousTradingState["previous_starfruit_prices"]
        else:
            self.previous_starfruit_prices = []

        # get the current price and append it to the list
        lowest_sell_price = sorted(tradingState.order_depths["STARFRUIT"].sell_orders.keys())[0]
        highest_buy_price = sorted(tradingState.order_depths["STARFRUIT"].buy_orders.keys(), reverse=True)[0]
        current_mid_price = (lowest_sell_price + highest_buy_price) / 2
        self.previous_starfruit_prices.append(current_mid_price)
        if len(self.previous_starfruit_prices) > 4:
            self.previous_starfruit_prices.pop(0)

    def predict_future_price(self):
        if len(self.prev_price) < self.linear_regression_ref_num:
            '''I modify here'''
            return "None"  # Not enough data
        future_price = self.regression_constant
        for i in range(len(self.theta)):
            future_price += self.theta[i] * self.prev_price[i]
        return future_price

    def get_acceptable_price(self, state: TradingState, product: str) -> int | float | None:
        if product == "AMETHYSTS":
            return 10000
        if product == "STARFRUIT":
            return self.get_starfruit_price()

    def get_starfruit_price(self) -> float | None:
        # if we don't have enough data, return None
        if len(self.previous_starfruit_prices) < 4:
            return None
        # calculate the average of the last four prices
        expected_price = 17.36384211 + sum([0.34608026, 0.26269948, 0.19565408, 0.19213413][i] * self.previous_starfruit_prices[i] for i in range(4))
        return expected_price

    def get_orders(self, state: TradingState, acceptable_price: int | float, product: str, price_aggression: int) -> List[Order]:
        product_order_depth = state.order_depths[product]
        product_position_limit = self.position_limit[product]
        acceptable_buy_price = math.floor(acceptable_price)
        acceptable_sell_price = math.ceil(acceptable_price)
        orders = []
        orders_sell = sorted(list(product_order_depth.sell_orders.items()), key=lambda x: x[0])
        orders_buy = sorted(list(product_order_depth.buy_orders.items()), key=lambda x: x[0], reverse=True)
        lowest_sell_price = orders_sell[0][0]
        lowest_buy_price = orders_buy[0][0]
        buying_pos = state.position.get(product, 0)
        logger.print(f"{product} current buying position: {buying_pos}")
        for ask, vol in orders_sell:
            if product_position_limit - buying_pos <= 0:
                break
            if ask < acceptable_price - price_aggression:
                buy_amount = min(-vol, product_position_limit - buying_pos)
                buying_pos += buy_amount
                #assert(buy_amount > 0)
                orders.append(Order(product, ask, buy_amount))
                logger.print(f"{product} buy order 1: {vol} at {ask}")
            if ask == acceptable_buy_price - price_aggression and buying_pos < 0:
                buy_amount = min(-vol, -buying_pos)
                buying_pos += buy_amount
                #assert(buy_amount > 0)
                orders.append(Order(product, ask, buy_amount))
                logger.print(f"{product} buy order 2: {vol} at {ask}")
        if product_position_limit - buying_pos > 0:
            if buying_pos < 0:  # overleveraged
                target_buy_price = min(acceptable_buy_price, lowest_buy_price + 1)
                vol = -buying_pos
                orders.append(Order(product, target_buy_price, vol))
                logger.print(f"{product} buy order 3: {vol} at {target_buy_price}")
                buying_pos += vol
            if 0 <= buying_pos < 10:  # slightly leveraged
                target_buy_price = min(acceptable_buy_price - 1, lowest_buy_price + 1)
                vol = 10 - buying_pos
                orders.append(Order(product, target_buy_price, vol))
                logger.print(f"{product} buy order 4: {vol} at {target_buy_price}")
                buying_pos += vol
            if buying_pos >= 10:  # neutral or better
                target_buy_price = min(acceptable_buy_price - 2, lowest_buy_price + 1)
                vol = product_position_limit - buying_pos
                orders.append(Order(product, target_buy_price, vol))
                logger.print(f"{product} buy order 5: {vol} at {target_buy_price}")
                buying_pos += vol
        selling_pos = state.position.get(product, 0)
        logger.print(f"{product} current selling position: {selling_pos}")
        for bid, vol in orders_buy:
            if -product_position_limit - selling_pos >= 0:
                break
            if bid > acceptable_price + price_aggression:
                sell_amount = max(-vol, -product_position_limit - selling_pos)
                selling_pos += sell_amount
                #assert(sell_amount < 0)
                orders.append(Order(product, bid, sell_amount))
                logger.print(f"{product} sell order 1: ", sell_amount, bid)
            if bid == acceptable_sell_price + price_aggression and selling_pos > 0:
                sell_amount = max(-vol, -selling_pos)
                selling_pos += sell_amount
                #assert(sell_amount < 0)
                orders.append(Order(product, bid, sell_amount))
                logger.print(f"{product} sell order 2: ", sell_amount, bid)
        if -product_position_limit - selling_pos < 0:
            if selling_pos > -0:
                target_sell_price = max(acceptable_sell_price, lowest_sell_price - 1)
                vol = -selling_pos
                orders.append(Order(product, target_sell_price, vol))
                selling_pos += vol
                logger.print(f"{product} sell order 3: selling {vol} at {target_sell_price}")
            if -0 >= selling_pos >= -10:
                target_sell_price = max(acceptable_sell_price + 1, lowest_sell_price - 1)
                vol = -selling_pos - 10
                orders.append(Order(product, target_sell_price, vol))
                selling_pos += vol
                logger.print(f"{product} sell order 4: selling {vol} at {target_sell_price}")
            if -10 >= selling_pos:
                target_sell_price = max(acceptable_sell_price + 2, lowest_sell_price - 1)
                vol = -product_position_limit - selling_pos
                orders.append(Order(product, target_sell_price, vol))
                selling_pos += vol
                logger.print(f"{product} sell order 5: selling {vol} at {target_sell_price}")
        return orders

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
                self.POSITION = previous_state.get('POSITION', {})
                # print("DECODING: ", self.last_prices)
            except:
                previous_state = {}
            self.update_starfruit_price_history(previous_state, state)

        for product, trades in state.market_trades.items():
            if trades:
                last_trade_price = trades[-1].price
                self.last_prices[product].append(last_trade_price)
                if len(self.last_prices[product]) > 100:
                    self.last_prices[product] = self.last_prices[product][-100:]
        
        # update position
        for product in state.position:
            self.POSITION[product] = state.position.get(product, 0)

        # print("traderData: " + state.traderData)
        # print(f"traderData: {self.predict_future_price()}" + state.traderData)
        # print("Observations: " + str(state.observations))

        result = {"STARFRUIT": [], "AMETHYSTS": [], "ORCHIDS": []}
        # orders: List[Order] = []
        traderData = ""
        conversions = 0

        # starfruit_orders, _ = self.algorithm_linear_regression("STARFRUIT", state)
        # amethysts_orders, _ = self.algorithm_mean_reversion("AMETHYSTS", state, 10000)

        # logger.print(f"STARFRUIT ORDERS: {starfruit_orders}")
        # logger.print(f"AMETHYSTS ORDERS: {amethysts_orders}")

        # result["STARFRUIT"] = starfruit_orders
        # result["AMETHYSTS"] = amethysts_orders

        # ORCHIDS
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orchids_orders = []
            product_acceptable_price = self.get_acceptable_price(state, product)
            if product == 'ORCHIDS':
                shipping_cost = state.observations.conversionObservations['ORCHIDS'].transportFees
                import_tariff = state.observations.conversionObservations['ORCHIDS'].importTariff
                export_tariff = state.observations.conversionObservations['ORCHIDS'].exportTariff
                ducks_ask = state.observations.conversionObservations['ORCHIDS'].askPrice
                ducks_bid = state.observations.conversionObservations['ORCHIDS'].bidPrice

                buy_from_ducks_prices = ducks_ask + shipping_cost + import_tariff
                # not used
                sell_to_ducks_prices = ducks_bid + shipping_cost + export_tariff

                lower_bound = int(round(buy_from_ducks_prices)) - 1
                upper_bound = int(round(buy_from_ducks_prices)) + 1

                logger.print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

                orchids_orders += self.calculate_orchids_orders(product, order_depth, lower_bound, upper_bound, orchids=True)
                conversions = -self.POSITION.get(product, 0)
                result["ORCHIDS"] = orchids_orders
            elif product == 'STARFRUIT':
                if product_acceptable_price is None:
                    continue
                starfruit_orders = self.get_orders(state, product_acceptable_price, product, PRICE_AGGRESSION[product])
                result["STARFRUIT"] = starfruit_orders
                print(f"STARFRUIT ORDERS: {starfruit_orders}")
            elif product == 'AMETHYSTS':
                if product_acceptable_price is None:
                    continue
                amethysts_orders = self.get_orders(state, product_acceptable_price, product, PRICE_AGGRESSION[product])
                result["AMETHYSTS"] = amethysts_orders
        
        logger.print(f"ORCHIDS CONVERSIONS: {conversions}")

        # encode
        traderData = jsonpickle.encode(
            {'last_prices': self.last_prices, 'POSITION': self.POSITION, 'previous_starfruit_prices': self.previous_starfruit_prices})
        # logger
        logger.flush(state, result, conversions, traderData)
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
            bid_price = int(round(bid_price))
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
            optimal_bid_price = lowest_bid_price
            orders.append(Order(symbol, int(round(optimal_bid_price)), max_bid_quantity))


        # Reset the current position
        current_position = state.position.get(symbol, 0)
        # Process selling opportunities if the price is above the mean plus the adaptive gap
        for ask_price, ask_quantity in symbol_buy_order_depth:
            ask_price = int(round(ask_price))
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
            optimal_ask_price = highest_ask_price
            orders.append(Order(symbol, int(round(optimal_ask_price)), max_ask_quantity))
        
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
    
    def get_volume_and_best_price(self, orders, buy_order):
        """
        Calculate the total volume and determine the best price for buy or sell orders.
        """
        INF = float('inf')
        prices = orders.keys()
        volumes = orders.values()

        if buy_order:
            total_volume = sum(volumes)
            best_price = max(prices, default=0)
        else:
            total_volume = -sum(volumes)
            best_price = min(prices, default=INF)

        return total_volume, best_price

    def calculate_orchids_orders(self, product, order_depth, our_bid_price, our_ask_price, orchids=False):
        orders: List[Order] = []

        sell_orders = dict(sorted(order_depth.sell_orders.items()))
        buy_orders = dict(sorted(order_depth.buy_orders.items(), reverse=True))

        _, best_sell_price = self.get_volume_and_best_price(sell_orders, buy_order=False)
        _, best_buy_price = self.get_volume_and_best_price(buy_orders, buy_order=True)

        position = self.POSITION[product] if not orchids else 0
        limit = self.position_limit[product]

        # TEST using dynamic gap - not working properly
        # dynamic_gap = self.get_dynamic_gap(product)
        # penny_buy = ceil(best_buy_price + dynamic_gap)
        # penny_sell = floor(best_sell_price - dynamic_gap)

        # penny the bid and ask
        penny_buy = best_buy_price + 1
        penny_sell = best_sell_price - 1
        # TESTING FOR ORCHIDS
        if orchids:
            penny_sell = best_sell_price - 5

        bid_price = min(our_bid_price, penny_buy)
        ask_price = max(our_ask_price, penny_sell)

        logger.print(f"Best buy price: {best_buy_price}, Best sell price: {best_sell_price}")
        # logger.print(f"Dynamic gap: {dynamic_gap}")
        logger.print(f"Penny buy: {penny_buy}, Penny sell: {penny_sell}")
        logger.print(f"Our bid price: {our_bid_price}, Our ask price: {our_ask_price}")

        # Market taking and making strategies
        # Buy orders
        for ask_sell, volume_sell in sell_orders.items():
            if position < limit and (ask_sell <= our_bid_price or (position < 0 and ask_sell == our_bid_price+1)): 
                num_to_sell = min(-volume_sell, limit - position)
                position += num_to_sell
                orders.append(Order(product, ask_sell, num_to_sell))

        # Market making by pennying
        if position < limit:
            num_to_sell = limit - position
            orders.append(Order(product, bid_price, num_to_sell))
            position += num_to_sell

        # reset position
        position = self.POSITION[product] if not orchids else 0

        # Market taking and making strategies
        for bid_buy, volume_buy in buy_orders.items():
            if position > -limit and (bid_buy >= our_ask_price or (position > 0 and bid_buy + 1 == our_ask_price)):
                num_to_sell = max(-volume_buy, -limit-position)
                position += num_to_sell
                orders.append(Order(product, bid_buy, num_to_sell))

        # Market making by pennying
        if position > -limit:
            num_to_sell = -limit - position
            orders.append(Order(product, ask_price, num_to_sell))
            position += num_to_sell 

        return orders
