from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np
import jsonpickle
from typing import List, Dict, Any
import json
import math
import pandas as pd

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
    "ORCHIDS": 0,
    'CHOCOLATE': 0,
    'STRAWBERRIES': 0,
    'ROSES': 0,
    'GIFT_BASKET': 0
}

class Trader:
    prev_price = []
    linear_regression_ref_num = 6
    previous_starfruit_prices = []

    def __init__(self) -> None:
        # position limit
        self.position_limit = {"STARFRUIT": 20, "AMETHYSTS": 20, "ORCHIDS": 100, "CHOCOLATE": 250,
            "STRAWBERRIES": 350,
            "ROSES": 60,
            "GIFT_BASKET": 60}
        
        # Initialize positions for all products
        self.POSITION = {product: 0 for product in self.position_limit.keys()}

        # Initialize price history for all products
        self.last_prices = {product: [] for product in self.position_limit.keys()}

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

        self.basket_mean_difference = 379.4904833333333
        self.basket_std = 76.42438217375009
        self.basket_std_factor = 0.70

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
        # state.traderData restore
        if state.traderData:
            try:
                previous_state = jsonpickle.decode(state.traderData)
                self.last_prices = previous_state.get('last_prices', {})
                self.POSITION = previous_state.get('POSITION', {})
            except:
                previous_state = {}
            self.update_starfruit_price_history(previous_state, state)

        for product, trades in state.market_trades.items():
            if trades:
                last_trade_price = trades[-1].price
                if product not in self.last_prices:
                    self.last_prices[product] = []
                self.last_prices[product].append(last_trade_price)
                if len(self.last_prices[product]) > 100:
                    self.last_prices[product] = self.last_prices[product][-100:]
        
        # update position
        for product in state.position:
            self.POSITION[product] = state.position.get(product, 0)

        result = {"STARFRUIT": [], "AMETHYSTS": [], "ORCHIDS": [], "GIFT_BASKET": [], "CHOCOLATE": [], "STRAWBERRIES": [], "ROSES": []}
        traderData = ""
        conversions = 0

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
            elif product == 'GIFT_BASKET':
                basket_orders = self.compute_basket_trades(state)
                result["GIFT_BASKET"] = basket_orders
            elif product in ['CHOCOLATE', 'STRAWBERRIES', 'ROSES']:
                model_orders = self.trade_based_on_model_predictions(state, product)
                result[product] = model_orders
        
        logger.print(f"ORCHIDS CONVERSIONS: {conversions}")

        # encode
        traderData = jsonpickle.encode(
            {'last_prices': self.last_prices, 'POSITION': self.POSITION, 'previous_starfruit_prices': self.previous_starfruit_prices})
        # logger
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def trade_based_on_model_predictions(self, state, product_name):
        current_prices = {
            'CHOCOLATE': self.get_current_price(state, 'CHOCOLATE'),
            'STRAWBERRIES': self.get_current_price(state, 'STRAWBERRIES'),
            'ROSES': self.get_current_price(state, 'ROSES'),
            'GIFT_BASKET': self.get_current_price(state, 'GIFT_BASKET')
        }

        current_prices = {k: v for k, v in current_prices.items() if v is not None}

        # Equations for each product
        predicted_prices = {
            'CHOCOLATE': 0.2045 * current_prices.get('GIFT_BASKET', 0) - 0.1525 * current_prices.get('ROSES', 0) - 1.8049 * current_prices.get('STRAWBERRIES', 0) + 2931.9884,
            'STRAWBERRIES': -0.3811 * current_prices.get('CHOCOLATE', 0) + 0.0807 * current_prices.get('GIFT_BASKET', 0) - 0.0698 * current_prices.get('ROSES', 0) + 2350.7398,
            'ROSES': -1.3445 * current_prices.get('CHOCOLATE', 0) + 0.5240 * current_prices.get('GIFT_BASKET', 0) - 2.9141 * current_prices.get('STRAWBERRIES', 0) - 169.2168
        }

        if product_name not in predicted_prices:
            print(f"No model available for {product_name}")
            return []

        predicted_price = predicted_prices[product_name]
        current_price = current_prices.get(product_name)

        if current_price is None:
            print(f"No current price available for {product_name}")
            return []

        if predicted_price > current_price:
            return [self.create_order(product_name, current_price, 'buy')]
        elif predicted_price < current_price:
            return [self.create_order(product_name, current_price, 'sell')]

        return []

    def create_order(self, product, price, action):
        quantity = min(self.position_limit[product], abs(self.POSITION[product]))
        if action == 'buy':
            order_price = math.floor(price)
            return Order(product, order_price, quantity)
        elif action == 'sell':
            order_price = math.ceil(price)
            return Order(product, order_price, -quantity)

    def compute_basket_trades(self, state):
        basket_products = ["GIFT_BASKET", "CHOCOLATE", "STRAWBERRIES", "ROSES"]
        mid_prices = {}

        for item in basket_products:
            _, best_sell_price = self.get_volume_and_best_price(state.order_depths[item].sell_orders, buy_order=False)
            _, best_buy_price = self.get_volume_and_best_price(state.order_depths[item].buy_orders, buy_order=True)
            mid_prices[item] = round((best_sell_price + best_buy_price) / 2)

        diff = mid_prices["GIFT_BASKET"] - (6 * mid_prices["STRAWBERRIES"] + 4 * mid_prices["CHOCOLATE"] + mid_prices["ROSES"]) - self.basket_mean_difference

        basket_orders = []
        order_depth = state.order_depths["GIFT_BASKET"]  # Assume GIFT_BASKET order depth is needed here

        worst_bid = min(order_depth.buy_orders.keys())
        worst_ask = max(order_depth.sell_orders.keys())

        if diff > self.basket_std_factor * self.basket_std:
            basket_orders += self.calculate_orchids_orders("GIFT_BASKET", order_depth, -int(1e9), worst_bid, gift_basket=True)
        elif diff < -self.basket_std_factor * self.basket_std:
            basket_orders += self.calculate_orchids_orders("GIFT_BASKET", order_depth, worst_ask, int(1e9), gift_basket=True)

        return basket_orders
    
    def get_current_price(self, state: TradingState, product):
        if product in state.order_depths:
            order_depth = state.order_depths[product]
            sell_orders = sorted(order_depth.sell_orders.items())
            buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
            if sell_orders and buy_orders:
                current_price = (sell_orders[0][0] + buy_orders[0][0]) / 2
                logger.print(f"Current price for {product}: {current_price}")
                return current_price
            else:
                logger.print(f"No sufficient order data for {product}. Sell orders: {len(sell_orders)}, Buy orders: {len(buy_orders)}")
        else:
            logger.print(f"No order depth found for {product}")
        return None

        
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

    def calculate_orchids_orders(self, product, order_depth, our_bid_price, our_ask_price, orchids=False, gift_basket=False):
        orders: List[Order] = []

        sell_orders = dict(sorted(order_depth.sell_orders.items()))
        buy_orders = dict(sorted(order_depth.buy_orders.items(), reverse=True))

        _, best_sell_price = self.get_volume_and_best_price(sell_orders, buy_order=False)
        _, best_buy_price = self.get_volume_and_best_price(buy_orders, buy_order=True)

        position = self.POSITION[product] if not orchids else 0
        limit = self.position_limit[product]

        penny_buy = best_buy_price + 1
        penny_sell = best_sell_price - 1

        if orchids:
            penny_sell = best_sell_price - 5

        bid_price = min(our_bid_price, penny_buy)
        ask_price = max(our_ask_price, penny_sell)

        if gift_basket:
            bid_price = our_bid_price
            ask_price = our_ask_price

        logger.print(f"Best buy price: {best_buy_price}, Best sell price: {best_sell_price}")
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
