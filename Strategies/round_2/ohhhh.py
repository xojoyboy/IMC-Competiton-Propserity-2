import jsonpickle
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
import math

PRICE_AGGRESSION = {
    "AMETHYSTS": 0,
    "STARFRUIT": 0,
    "ORCHIDS": 0
}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))
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

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])
        return compressed

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [observation.bidPrice, observation.askPrice, observation.transportFees, observation.exportTariff, observation.importTariff, observation.sunlight, observation.humidity]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> list[list[Any]]:
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

class Trader:
    POSITION_LIMITS = {"AMETHYSTS": 20, "STARFRUIT": 20, "ORCHIDS": 20}
    previous_starfruit_prices = []

    def __init__(self) -> None:
        self.logger = Logger()

        self.POSITION = {}

        self.POSITION['AMETHYSTS'] = 0
        self.POSITION['STARFRUIT'] = 0
        self.POSITION['ORCHIDS'] = 0

        self.position_limit = {"STARFRUIT": 20, "AMETHYSTS": 20, "ORCHIDS": 20}

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

    def get_starfruit_price(self) -> float | None:
        # if we don't have enough data, return None
        if len(self.previous_starfruit_prices) < 4:
            return None
        # calculate the average of the last four prices
        expected_price = 17.36384211 + sum([0.34608026, 0.26269948, 0.19565408, 0.19213413][i] * self.previous_starfruit_prices[i] for i in range(4))
        return expected_price

    def get_orders(self, state: TradingState, acceptable_price: int | float, product: str, price_aggression: int) -> List[Order]:
        product_order_depth = state.order_depths[product]
        product_position_limit = self.POSITION_LIMITS[product]
        acceptable_buy_price = math.floor(acceptable_price)
        acceptable_sell_price = math.ceil(acceptable_price)
        orders = []
        orders_sell = sorted(list(product_order_depth.sell_orders.items()), key=lambda x: x[0])
        orders_buy = sorted(list(product_order_depth.buy_orders.items()), key=lambda x: x[0], reverse=True)
        lowest_sell_price = orders_sell[0][0]
        lowest_buy_price = orders_buy[0][0]
        buying_pos = state.position.get(product, 0)
        self.logger.print(f"{product} current buying position: {buying_pos}")
        for ask, vol in orders_sell:
            if product_position_limit - buying_pos <= 0:
                break
            if ask < acceptable_price - price_aggression:
                buy_amount = min(-vol, product_position_limit - buying_pos)
                buying_pos += buy_amount
                #assert(buy_amount > 0)
                orders.append(Order(product, ask, buy_amount))
                self.logger.print(f"{product} buy order 1: {vol} at {ask}")
            if ask == acceptable_buy_price - price_aggression and buying_pos < 0:
                buy_amount = min(-vol, -buying_pos)
                buying_pos += buy_amount
                #assert(buy_amount > 0)
                orders.append(Order(product, ask, buy_amount))
                self.logger.print(f"{product} buy order 2: {vol} at {ask}")
        if product_position_limit - buying_pos > 0:
            if buying_pos < 0:  # overleveraged
                target_buy_price = min(acceptable_buy_price, lowest_buy_price + 1)
                vol = -buying_pos
                orders.append(Order(product, target_buy_price, vol))
                self.logger.print(f"{product} buy order 3: {vol} at {target_buy_price}")
                buying_pos += vol
            if 0 <= buying_pos < 10:  # slightly leveraged
                target_buy_price = min(acceptable_buy_price - 1, lowest_buy_price + 1)
                vol = 10 - buying_pos
                orders.append(Order(product, target_buy_price, vol))
                self.logger.print(f"{product} buy order 4: {vol} at {target_buy_price}")
                buying_pos += vol
            if buying_pos >= 10:  # neutral or better
                target_buy_price = min(acceptable_buy_price - 2, lowest_buy_price + 1)
                vol = product_position_limit - buying_pos
                orders.append(Order(product, target_buy_price, vol))
                self.logger.print(f"{product} buy order 5: {vol} at {target_buy_price}")
                buying_pos += vol
        selling_pos = state.position.get(product, 0)
        self.logger.print(f"{product} current selling position: {selling_pos}")
        for bid, vol in orders_buy:
            if -product_position_limit - selling_pos >= 0:
                break
            if bid > acceptable_price + price_aggression:
                sell_amount = max(-vol, -product_position_limit - selling_pos)
                selling_pos += sell_amount
                #assert(sell_amount < 0)
                orders.append(Order(product, bid, sell_amount))
                self.logger.print(f"{product} sell order 1: ", sell_amount, bid)
            if bid == acceptable_sell_price + price_aggression and selling_pos > 0:
                sell_amount = max(-vol, -selling_pos)
                selling_pos += sell_amount
                #assert(sell_amount < 0)
                orders.append(Order(product, bid, sell_amount))
                self.logger.print(f"{product} sell order 2: ", sell_amount, bid)
        if -product_position_limit - selling_pos < 0:
            if selling_pos > -0:
                target_sell_price = max(acceptable_sell_price, lowest_sell_price - 1)
                vol = -selling_pos
                orders.append(Order(product, target_sell_price, vol))
                selling_pos += vol
                self.logger.print(f"{product} sell order 3: selling {vol} at {target_sell_price}")
            if -0 >= selling_pos >= -10:
                target_sell_price = max(acceptable_sell_price + 1, lowest_sell_price - 1)
                vol = -selling_pos - 10
                orders.append(Order(product, target_sell_price, vol))
                selling_pos += vol
                self.logger.print(f"{product} sell order 4: selling {vol} at {target_sell_price}")
            if -10 >= selling_pos:
                target_sell_price = max(acceptable_sell_price + 2, lowest_sell_price - 1)
                vol = -product_position_limit - selling_pos
                orders.append(Order(product, target_sell_price, vol))
                selling_pos += vol
                self.logger.print(f"{product} sell order 5: selling {vol} at {target_sell_price}")
        return orders

    def get_acceptable_price(self, state: TradingState, product: str) -> int | float | None:
        if product == "AMETHYSTS":
            return 10000
        if product == "STARFRUIT":
            return self.get_starfruit_price()
        return None

    def run(self, state: TradingState):
        try:
            previousStateData = jsonpickle.decode(state.traderData)
        except:
            previousStateData = {}
        self.update_starfruit_price_history(previousStateData, state)

        result = {}
        
        for product in state.order_depths:
            order_depth = state.order_depths[product]
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

                self.logger.print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

                orchids_orders = self.calculate_orchids_orders(product, order_depth, lower_bound, upper_bound, orchids=True)
                result["ORCHIDS"] = orchids_orders
                conversions = -self.POSITION.get(product, 0)
            else:
                if product_acceptable_price is None:
                    continue
                orders = self.get_orders(state, product_acceptable_price, product, PRICE_AGGRESSION[product])
                result[product] = orders

        traderData = {
            "previous_starfruit_prices": self.previous_starfruit_prices
        } 

        serialisedTraderData = jsonpickle.encode(traderData)
        if serialisedTraderData == None:
            serialisedTraderData = ""

        conversions = 0  # Not really documented in the task description
        self.logger.flush(state, result, conversions, serialisedTraderData)

        return result, conversions, serialisedTraderData

    def calculate_orchids_orders(self, product, order_depth, our_bid_price, our_ask_price, orchids=False):
        orders: List[Order] = []
        sell_orders = dict(sorted(order_depth.sell_orders.items()))
        buy_orders = dict(sorted(order_depth.buy_orders.items(), reverse=True))
        _, best_sell_price = self.get_volume_and_best_price(sell_orders, buy_order=False)
        _, best_buy_price = self.get_volume_and_best_price(buy_orders, buy_order=True)
        position = self.POSITION[product] if not orchids else 0
        limit = self.POSITION_LIMITS[product]
        penny_buy = best_buy_price + 1
        penny_sell = best_sell_price - 1
        if orchids:
            penny_sell = best_sell_price - 5
        bid_price = min(our_bid_price, penny_buy)
        ask_price = max(our_ask_price, penny_sell)
        self.logger.print(f"Best buy price: {best_buy_price}, Best sell price: {best_sell_price}")
        self.logger.print(f"Penny buy: {penny_buy}, Penny sell: {penny_sell}")
        self.logger.print(f"Our bid price: {our_bid_price}, Our ask price: {our_ask_price}")
        for ask_sell, volume_sell in sell_orders.items():
            if position < limit and (ask_sell <= our_bid_price or (position < 0 and ask_sell == our_bid_price+1)):
                num_to_sell = min(-volume_sell, limit - position)
                position += num_to_sell
                orders.append(Order(product, ask_sell, num_to_sell))
        if position < limit:
            num_to_sell = limit - position
            orders.append(Order(product, bid_price, num_to_sell))
            position += num_to_sell
        position = self.POSITION[product] if not orchids else 0
        for bid_buy, volume_buy in buy_orders.items():
            if position > -limit and (bid_buy >= our_ask_price or (position > 0 and bid_buy + 1 == our_ask_price)):
                num_to_sell = max(-volume_buy, -limit-position)
                position += num_to_sell
                orders.append(Order(product, bid_buy, num_to_sell))
        if position > -limit:
            num_to_sell = -limit - position
            orders.append(Order(product, ask_price, num_to_sell))
            position += num_to_sell
        return orders

    def get_volume_and_best_price(self, orders, buy_order):
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
