from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np
import jsonpickle
from typing import List, Dict, Any
import json
import math
import statistics

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
    'ROSES': 30,
    'GIFT_BASKET': 20
}

class Trader:
    previous_starfruit_prices = []

    etf_returns = []
    assets_returns = []
    chocolate_returns = []
    chocolate_estimated_returns = []
    strawberries_returns = []
    strawberries_estimated_returns = []

    coconut_coupon_returns = []
    coconut_coupon_bsm_returns = []
    coconut_returns = []
    coconut_estimated_returns = []

    rhianna_buy = False
    rhianna_trade_before = False

    N = statistics.NormalDist(mu=0, sigma=1)

    def __init__(self) -> None:
        # position limit
        self.position_limit = {"STARFRUIT": 20, "AMETHYSTS": 20, "ORCHIDS": 100, "CHOCOLATE": 250,
            "STRAWBERRIES": 350,
            "ROSES": 60,
            "GIFT_BASKET": 60,
            "COCONUT": 300,
            "COCONUT_COUPON": 600}

        # Initialize positions for all products
        self.POSITION = {product: 0 for product in self.position_limit.keys()}

    def update_starfruit_price_history(self, previousTradingState, tradingState: TradingState):
        self.previous_starfruit_prices = previousTradingState.get("previous_starfruit_prices", [])

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
        return None

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

    # STRAWBERRIES
    def compute_strawberries_orders(self, state: TradingState):
        products = ["STRAWBERRIES"]
        positions, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "STRAWBERRIES": []}

        for product in products:
            positions[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        self.strawberries_returns.append(prices["STRAWBERRIES"])

        if len(self.strawberries_returns) < 100:
            return orders

        # Slow moving average
        strawberries_rolling_mean = statistics.fmean(self.strawberries_returns[-200:])
        # Fast moving average
        strawberries_rolling_mean_fast = statistics.fmean(self.strawberries_returns[-100:])

        # Empirically tuned to avoid noisy buy and sell signals - do nothing if sideways market
        if strawberries_rolling_mean_fast > strawberries_rolling_mean + 1.5:

            # Fixed entry every timestep that criteria is met, max-ing out early
            limit_mult = 18
            limit_mult = min(limit_mult, self.position_limit["STRAWBERRIES"] - positions["STRAWBERRIES"], self.position_limit["STRAWBERRIES"])
            orders["STRAWBERRIES"].append(Order("STRAWBERRIES", best_asks["STRAWBERRIES"], limit_mult))

        elif strawberries_rolling_mean_fast < strawberries_rolling_mean - 1.5:

            # Fixed entry every timestep, max-ing out early
            limit_mult = -18
            limit_mult = max(limit_mult, -self.position_limit["STRAWBERRIES"] - positions["STRAWBERRIES"], -self.position_limit["STRAWBERRIES"])
            orders["STRAWBERRIES"].append(Order("STRAWBERRIES", best_bids["STRAWBERRIES"], limit_mult))

        return orders

    # CHOCOLATE
    def compute_chocolate_orders(self, state: TradingState):
        products = ["CHOCOLATE"]
        positions, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "CHOCOLATE": []}

        for product in products:
            positions[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        self.chocolate_returns.append(prices["CHOCOLATE"])

        if len(self.chocolate_returns) < 100:
            return orders

        # Slow moving average
        chocolate_rolling_mean = statistics.fmean(self.chocolate_returns[-200:])
        # Fast moving average
        chocolate_rolling_mean_fast = statistics.fmean(self.chocolate_returns[-100:])

        # Empirically tuned to avoid noisy buy and sell signals - do nothing if sideways market
        if chocolate_rolling_mean_fast > chocolate_rolling_mean + 1.5:

            # Fixed entry every timestep that criteria is met, max-ing out early
            limit_mult = 11

            limit_mult = min(limit_mult, self.position_limit["CHOCOLATE"] - positions["CHOCOLATE"], self.position_limit["CHOCOLATE"])
            orders["CHOCOLATE"].append(Order("CHOCOLATE", best_asks["CHOCOLATE"], limit_mult))

        elif chocolate_rolling_mean_fast < chocolate_rolling_mean - 1.5:

            # Fixed entry every timestep, max-ing out early
            limit_mult = -11
            limit_mult = max(limit_mult, -self.position_limit["CHOCOLATE"] - positions["CHOCOLATE"], -self.position_limit["CHOCOLATE"])

            orders["CHOCOLATE"].append(Order("CHOCOLATE", best_bids["CHOCOLATE"], limit_mult))

        return orders

    # ROSE
    def compute_rose_orders(self, state: TradingState):
        orders = []

        roses_pos = state.position["ROSES"] if "ROSES" in state.position else 0
        best_bid = max(state.order_depths["ROSES"].buy_orders.keys())
        bid_vol = state.order_depths["ROSES"].buy_orders[best_bid]
        best_ask = min(state.order_depths["ROSES"].sell_orders.keys())
        ask_vol = state.order_depths["ROSES"].sell_orders[best_ask]

        if "ROSES" not in state.market_trades:
            return orders

        for trade in state.market_trades["ROSES"]:
            if trade.buyer == "Rhianna":
                self.rhianna_buy = True
                self.rhianna_trade_before = True
            elif trade.seller == "Rhianna":
                self.rhianna_buy = False
                self.rhianna_trade_before = True

            # Buy signal
            if self.rhianna_buy:
                vol = max(-bid_vol, -self.position_limit["ROSES"] - min(0, roses_pos))
                orders.append(Order("ROSES", best_bid, vol))
                self.rhianna_buy = False
            # Sell signal
            elif self.rhianna_trade_before:
                vol = min(-ask_vol, self.position_limit["ROSES"] - max(0, roses_pos))
                orders.append(Order("ROSES", best_ask, vol))
                self.rhianna_buy = True

        return orders

    # NEW BASKET
    def compute_basket_orders(self, state: TradingState):
        products = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        positions, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "CHOCOLATE": [], "STRAWBERRIES": [], "ROSES": [], "GIFT_BASKET": []}
        
        for product in products:
            positions[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0
        
        estimated_price = 4.0 * prices['CHOCOLATE'] + 6.0 * prices['STRAWBERRIES'] + prices['ROSES']

        price_diff = prices["GIFT_BASKET"] - estimated_price

        self.etf_returns.append(prices["GIFT_BASKET"])
        self.assets_returns.append(price_diff)

        if len(self.etf_returns) < 100 or len(self.assets_returns) < 100:
            return orders
        
        # slow moving average
        assets_rolling_mean = statistics.fmean(self.assets_returns[-200:])
        # fast moving average
        assets_rolling_mean_fast = statistics.fmean(self.assets_returns[-100:])

        if assets_rolling_mean_fast > assets_rolling_mean + 4:

            # Fixed entry every timestep that criteria is met, max-ing out early
            limit_mult = 3

            limit_mult = min(limit_mult, self.position_limit["GIFT_BASKET"] - positions["GIFT_BASKET"], self.position_limit["GIFT_BASKET"])
            orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_asks["GIFT_BASKET"], limit_mult))
        elif assets_rolling_mean_fast < assets_rolling_mean - 4:

            # Fixed entry every timestep, max-ing out early
            limit_mult = -3

            limit_mult = max(limit_mult, -self.position_limit["GIFT_BASKET"] - positions["GIFT_BASKET"], -self.position_limit["GIFT_BASKET"])
            orders["GIFT_BASKET"].append(Order("GIFT_BASKET", best_bids["GIFT_BASKET"], limit_mult))

        return orders

    # NEW COCONUT
    def compute_coconut_orders(self, state: TradingState):
        products = ["COCONUT"]
        positions, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {
            "COCONUT": []}

        for product in products:
            positions[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        self.coconut_returns.append(prices["COCONUT"])

        if len(self.coconut_returns) < 100:
            return orders

        # Slow moving average
        coconut_rolling_mean = statistics.fmean(self.coconut_returns[-200:])
        # Fast moving average
        coconut_rolling_mean_fast = statistics.fmean(self.coconut_returns[-100:])

        # Empirically tuned to avoid noisy buy and sell signals - do nothing if sideways market
        if coconut_rolling_mean_fast > coconut_rolling_mean + 4:

            # Fixed entry every timestep that criteria is met, max-ing out early
            limit_mult = 30
            limit_mult = min(limit_mult, self.position_limit["COCONUT"] - positions["COCONUT"], self.position_limit["COCONUT"])
            orders["COCONUT"].append(Order("COCONUT", best_asks["COCONUT"], limit_mult))

        elif coconut_rolling_mean_fast < coconut_rolling_mean - 4:

            # Fixed entry every timestep, max-ing out early
            limit_mult = -30
            limit_mult = max(limit_mult, -self.position_limit["COCONUT"] - positions["COCONUT"], -self.position_limit["COCONUT"])
            orders["COCONUT"].append(Order("COCONUT", best_bids["COCONUT"], limit_mult))

        return orders

    def BS_CALL(self, S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * self.N.cdf(d1) - K * np.exp(-r * T) * self.N.cdf(d2)

    # NEW COCONUT COUPON
    def compute_coconut_coupon_orders(self, state: TradingState):
        products = ["COCONUT_COUPON", "COCONUT"]
        positions, buy_orders, sell_orders, best_bids, best_asks, prices, orders = {}, {}, {}, {}, {}, {}, {"COCONUT_COUPON": [], "COCONUT": []}

        for product in products:
            positions[product] = state.position[product] if product in state.position else 0

            buy_orders[product] = state.order_depths[product].buy_orders
            sell_orders[product] = state.order_depths[product].sell_orders

            best_bids[product] = max(buy_orders[product].keys())
            best_asks[product] = min(sell_orders[product].keys())

            prices[product] = (best_bids[product] + best_asks[product]) / 2.0

        # Use BSM
        S = prices["COCONUT"]
        K = 10000
        T = 250
        r = 0
        sigma = 0.01011932923
        bsm_price = self.BS_CALL(S, K, T, r, sigma)

        self.coconut_coupon_returns.append(prices["COCONUT_COUPON"])
        self.coconut_coupon_bsm_returns.append(bsm_price)

        # Dummy for now
        self.coconut_returns.append(prices["COCONUT"])
        self.coconut_estimated_returns.append(prices["COCONUT"])

        if len(self.coconut_coupon_returns) < 2 or len(self.coconut_coupon_bsm_returns) < 2:
            return orders

        coconut_coupon_rolling_mean = statistics.fmean(self.coconut_coupon_returns[-200:])
        coconut_coupon_rolling_std = statistics.stdev(self.coconut_coupon_returns[-200:])

        coconut_coupon_bsm_rolling_mean = statistics.fmean(self.coconut_coupon_bsm_returns[-200:])
        coconut_coupon_bsm_rolling_std = statistics.stdev(self.coconut_coupon_bsm_returns[-200:])

        if coconut_coupon_rolling_std != 0:
            coconut_coupon_z_score = (self.coconut_coupon_returns[-1] - coconut_coupon_rolling_mean) / coconut_coupon_rolling_std
        else:
            coconut_coupon_z_score = 0

        if coconut_coupon_bsm_rolling_std != 0:
            coconut_coupon_bsm_z_score = (self.coconut_coupon_bsm_returns[-1] - coconut_coupon_bsm_rolling_mean) / coconut_coupon_bsm_rolling_std
        else:
            coconut_coupon_bsm_z_score = 0

        # May need a catch here to set both == 0 if one or the other is 0, to avoid errorneous z scores
        coconut_coupon_z_score_diff = coconut_coupon_z_score - coconut_coupon_bsm_z_score

        # Option is underpriced
        if coconut_coupon_z_score_diff < -1.2:
            coconut_coupon_best_ask_vol = sell_orders["COCONUT_COUPON"][best_asks["COCONUT_COUPON"]]

            limit_mult = -coconut_coupon_best_ask_vol
            limit_mult = round(limit_mult * abs(coconut_coupon_z_score_diff) / 2)
            limit_mult = min(limit_mult, self.position_limit["COCONUT_COUPON"] - positions["COCONUT_COUPON"], self.position_limit["COCONUT_COUPON"])
            orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_asks["COCONUT_COUPON"], limit_mult))

        # Option is overpriced
        elif coconut_coupon_z_score_diff > 1.2:
            coconut_coupon_best_bid_vol = buy_orders["COCONUT_COUPON"][best_bids["COCONUT_COUPON"]]

            limit_mult = coconut_coupon_best_bid_vol
            limit_mult = round(-limit_mult * abs(coconut_coupon_z_score_diff) / 2)
            limit_mult = max(limit_mult, -self.position_limit["COCONUT_COUPON"] - positions["COCONUT_COUPON"], -self.position_limit["COCONUT_COUPON"])
            orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_bids["COCONUT_COUPON"], limit_mult))

        return orders

    def refresh_runner_state(self, state: TradingState):
        try:
            previous_state = jsonpickle.decode(state.traderData)
            self.POSITION = previous_state.get('POSITION', {})
            self.update_starfruit_price_history(previous_state, state)
            # NEW BASKET
            self.etf_returns = previous_state.get('etf_returns', [])
            self.assets_returns = previous_state.get('assets_returns', [])
            # NEW ROSE
            self.rhianna_buy = previous_state.get('rhianna_buy', False)
            self.rhianna_trade_before = previous_state.get('rhianna_trade_before', False)
            # CHOCOLATE
            self.chocolate_returns = previous_state.get('chocolate_returns', [])
            self.chocolate_estimated_returns = previous_state.get('chocolate_estimated_returns', [])
            # STRAWBERRY
            self.strawberries_returns = previous_state.get('strawberries_returns', [])
            self.strawberries_estimated_returns = previous_state.get('strawberries_estimated_returns', [])
        except:
            pass

    def run(self, state: TradingState):
        # state.traderData restore
        self.refresh_runner_state(state)

        # update position
        for product in state.position:
            self.POSITION[product] = state.position.get(product, 0)

        result = {"STARFRUIT": [], "AMETHYSTS": [], "ORCHIDS": [], "GIFT_BASKET": [], "CHOCOLATE": [], "STRAWBERRIES": [], "ROSES": [], "COCONUT": [], "COCONUT_COUPON": []}
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

                lower_bound = int(round(buy_from_ducks_prices))
                upper_bound = int(round(sell_to_ducks_prices))

                logger.print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

                # orchids_orders += self.calculate_orchids_orders(product, order_depth, lower_bound, upper_bound, orchids=True)
                # conversions = -self.POSITION.get(product, 0)
                # result["ORCHIDS"] = orchids_orders
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

        # NEW BASKET
        basket_orders = self.compute_basket_orders(state)
        for product, orders in basket_orders.items():
            result[product] = orders

        # NEW ROSE
        rose_orders = self.compute_rose_orders(state)
        result["ROSES"] = rose_orders

        # NEW CHOCOLATE
        chocolate_orders = self.compute_chocolate_orders(state)
        for product, orders in chocolate_orders.items():
            result[product] = orders
        
        # NEW STRAWBERRY
        strawberries_orders = self.compute_strawberries_orders(state)
        for product, orders in strawberries_orders.items():
            result[product] = orders

        # NEW COCONUT COUPON
        coconut_coupon_orders = self.compute_coconut_coupon_orders(state)
        for product, orders in coconut_coupon_orders.items():
            result[product] = orders

        # NEW COCONUT
        coconut_orders = self.compute_coconut_orders(state)
        for product, orders in coconut_orders.items():
            result[product] = orders

        # encode
        data_to_encode = {
            'POSITION': self.POSITION,
            'previous_starfruit_prices': self.previous_starfruit_prices,
            'etf_returns': self.etf_returns,
            'assets_returns': self.assets_returns,
            'rhianna_buy': self.rhianna_buy,
            'rhianna_trade_before': self.rhianna_trade_before,
            'chocolate_returns': self.chocolate_returns,
            'chocolate_estimated_returns': self.chocolate_estimated_returns,
            'strawberries_returns': self.strawberries_returns,
            'strawberries_estimated_returns': self.strawberries_estimated_returns
        }
        traderData = jsonpickle.encode(data_to_encode)
        if traderData == None:
            traderData = ""

        # logger
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

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

        penny_buy = best_buy_price + 1
        penny_sell = best_sell_price - 1

        # if orchids:
        #     penny_sell = best_sell_price - 5

        bid_price = min(our_bid_price, penny_buy)
        ask_price = max(our_ask_price, penny_sell)

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
