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

GIFT_BASKET_COEFFICIENTS = [289.5698892751825, 0.8626389049176026, 0.031684672550904125, 
							0.008742473436567089, -0.01659117908903074, -0.010649786829042718, 
							-0.03995403832399447, 0.05569695953385345, 0.10981863478227183]
# for roses - goes 1 intercept, 5 chocolates, 5 strawberry coefficients, 5 gift basket
ROSES_COEFFICIENTS = [6676.023412674642, -0.6089558593367816, -0.052054262119180406, 0.011739688287736971, 
					  0.008602475083543615, 0.6826137408688111, -0.8974486405018114, -0.2723245494965232, 
					  -0.030985780039239685, -0.17674612974408843, -0.3651901445828294, 0.14785332793575073, 
					  0.00899070703038854, 0.0003905643176334017, 0.006529617716453329, 0.040845254460841716]
STRAWBERRY_COEFFICIENTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

THRESHOLDS = {
	"AMETHYSTS": {
		"over": 0,
		"mid": 10
	},
	"STARFRUIT": {
		"over": 0,
		"mid": 10
	},
	"ORCHIDS": {
		"over": 20,
		"mid": 40
	},
	"GIFT_BASKET": {
		"over": 0,
		"mid": 10
	},
	"ROSES": {
		"over": 0,
		"mid": 10
	},
}

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
    linear_regression_ref_num = 6
    previous_starfruit_prices = []
    previous_gift_basket_prices = []
    previous_chocolate_prices = []
    previous_strawberry_prices = []
    previous_rose_prices = []
    previous_premium_basket_prices = []
    market_taking: list[tuple[str, int, bool]] = []
    next_market_taking: list[tuple[str, int]] = []

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

        # COCONUT
        self.PREV_COCONUT_PRICE = -1
        self.PREV_COUPON_PRICE = -1
        self.COUPON_Z_SCORE = 1.2
        self.COUPON_IV_STORE = []
        self.COUPON_IV_STORE_SIZE = 100

    def update_starfruit_price_history(self, previousTradingState, tradingState: TradingState):
        self.previous_starfruit_prices = previousTradingState.get("previous_starfruit_prices", [])

        # get the current price and append it to the list
        lowest_sell_price = sorted(tradingState.order_depths["STARFRUIT"].sell_orders.keys())[0]
        highest_buy_price = sorted(tradingState.order_depths["STARFRUIT"].buy_orders.keys(), reverse=True)[0]

        current_mid_price = (lowest_sell_price + highest_buy_price) / 2

        self.previous_starfruit_prices.append(current_mid_price)

        if len(self.previous_starfruit_prices) > 4:
            self.previous_starfruit_prices.pop(0)
    
    def update_conversions(self, previousStateData, state: TradingState):
        self.market_taking = previousStateData.get("market_taking", [])

		# remove all market taking that has been seen
        self.market_taking = [(product, amount, seen) for product, amount, seen in self.market_taking if not seen]

    def update_combined_gift_basket_price_history(self, previousTradingState, tradingState: TradingState):
        self.previous_chocolate_prices = previousTradingState.get("previous_chocolate_prices", [])
        self.previous_rose_prices = previousTradingState.get("previous_rose_prices", [])
        self.previous_strawberry_prices = previousTradingState.get("previous_strawberry_prices", [])
        self.previous_gift_basket_prices = previousTradingState.get("previous_gift_basket_prices", [])
        self.previous_premium_basket_prices = previousTradingState.get("previous_premium_basket_prices", [])

        good_to_list = {
            "CHOCOLATE": self.previous_chocolate_prices,
            "STRAWBERRIES": self.previous_strawberry_prices,
            "ROSES": self.previous_rose_prices,
            "GIFT_BASKET": self.previous_premium_basket_prices
		}

		# get the current price and append it to the list
        gift_basket = 0

        for good in good_to_list:
            lowest_sell_price = sorted(tradingState.order_depths[good].sell_orders.keys())[0]
            highest_buy_price = sorted(tradingState.order_depths[good].buy_orders.keys(), reverse=True)[0]

            current_mid_price = (lowest_sell_price + highest_buy_price) / 2

            if good == "CHOCOLATE":
                gift_basket += current_mid_price * 4
            elif good == "STRAWBERRIES":
                gift_basket += current_mid_price * 6
            elif good == "ROSES":
                gift_basket += current_mid_price
		
            good_to_list[good].append(current_mid_price)

            if len(good_to_list[good]) > 5:
                good_to_list[good].pop(0)
		
        self.previous_gift_basket_prices.append(gift_basket)

        if len(self.previous_gift_basket_prices) > len(GIFT_BASKET_COEFFICIENTS) - 1:
            self.previous_gift_basket_prices.pop(0)

    def get_combined_gift_basket_price(self) -> float | None:
        # if we don't have enough data, return None
        if len(self.previous_gift_basket_prices) < len(GIFT_BASKET_COEFFICIENTS) - 1:
            return None
	
        expected_price = GIFT_BASKET_COEFFICIENTS[0] + sum([GIFT_BASKET_COEFFICIENTS[i + 1] * self.previous_gift_basket_prices[i] for i in range(len(GIFT_BASKET_COEFFICIENTS) - 1)])

        return expected_price

    def get_rose_price(self) -> float | None:
        CHOCOLATE_COEFFICIENTS = 5
        STRAWBERRY_COEFFICIENTS = 5
        GIFT_BASKET_COEFFICIENTS = 5

        if len(self.previous_chocolate_prices) < CHOCOLATE_COEFFICIENTS or len(self.previous_strawberry_prices) < STRAWBERRY_COEFFICIENTS:
            return None
		
        component_list = ([1.0] + self.previous_chocolate_prices[:CHOCOLATE_COEFFICIENTS] 
					+ self.previous_strawberry_prices[:STRAWBERRY_COEFFICIENTS]
					+ self.previous_gift_basket_prices[:GIFT_BASKET_COEFFICIENTS])

        expected_price = sum([ROSES_COEFFICIENTS[i] * component_list[i] for i in range(len(ROSES_COEFFICIENTS))])

        return expected_price

    def get_acceptable_price(self, state: TradingState, product: str) -> int | float | None:
        if product == "AMETHYSTS":
            return 10000
        if product == "STARFRUIT":
            return self.get_starfruit_price()
        if product == "GIFT_BASKET":
            return self.get_combined_gift_basket_price()
        if product == "ROSES":
            return self.get_rose_price()
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

    def get_orders_new(self, state: TradingState, acceptable_sell_price: int, acceptable_buy_price: int, product: str, price_aggression: int) -> List[Order]:
		# market taking + making based on Stanford's 2023 entry
        product_order_depth = state.order_depths[product]
        product_position_limit = self.position_limit[product]
        orders = []

        # sort the order books by price (will sort by the key by default)
        orders_sell = sorted(list(product_order_depth.sell_orders.items()), key = lambda x: x[0])
        orders_buy = sorted(list(product_order_depth.buy_orders.items()), key=lambda x: x[0], reverse=True)

        lowest_sell_price = orders_sell[0][0]
        lowest_buy_price = orders_buy[0][0]

        # we start with buying - using our current position to determine how much and how aggressively we buy from the market

        buying_pos = state.position.get(product, 0)

        for ask, vol in orders_sell:
			# skip if there is no quota left
            if product_position_limit - buying_pos <= 0:
                break

            if ask < acceptable_buy_price - price_aggression:
				# we want to buy
                buy_amount = min(-vol, product_position_limit - buying_pos)
                buying_pos += buy_amount
                assert(buy_amount > 0)
                orders.append(Order(product, ask, buy_amount))
                self.market_taking.append((product, buy_amount, False))

			# if overleveraged, buy up until we are no longer leveraged
            if ask == acceptable_buy_price - price_aggression and buying_pos < 0:
                buy_amount = min(-vol, -buying_pos)
                buying_pos += buy_amount
                assert(buy_amount > 0)
                orders.append(Order(product, ask, buy_amount))
                self.market_taking.append((product, buy_amount, False))

		# once we exhaust all profitable sell orders, we place additional buy orders
		# at a price acceptable to us
		# what that price looks like will depend on our position
		
        if product_position_limit - buying_pos > 0: # if we have capacity
            if buying_pos < THRESHOLDS[product]["over"]: # if we are overleveraged to sell, buy at parity for price up to neutral position
                target_buy_price = min(acceptable_buy_price - price_aggression, lowest_buy_price + 1)
                vol = -buying_pos + THRESHOLDS[product]["over"]
                orders.append(Order(product, target_buy_price, vol))
                buying_pos += vol
            if THRESHOLDS[product]["over"] <= buying_pos <= THRESHOLDS[product]["mid"]:
                target_buy_price = min(acceptable_buy_price - 1 - price_aggression, lowest_buy_price + 1)
                vol = -buying_pos + THRESHOLDS[product]["mid"] # if we are close to neutral
                orders.append(Order(product, target_buy_price, vol))
                buying_pos += vol
            if buying_pos >= THRESHOLDS[product]["mid"]:
                target_buy_price = min(acceptable_buy_price - 2 - price_aggression, lowest_buy_price + 1)
                vol = product_position_limit - buying_pos
                orders.append(Order(product, target_buy_price, vol))
                buying_pos += vol
				
		# now we sell - we reset our position
        selling_pos = state.position.get(product, 0)

        for bid, vol in orders_buy:
			# positive orders in the list
			# but we are sending negative sell orders, so we negate it
			# max we can sell is -product_position_limit - current position
			# if current position is negative we can sell less - if positive we can sell more
			
            if -product_position_limit - selling_pos >= 0:
                break

            if bid > acceptable_sell_price + price_aggression:
                sell_amount = max(-vol, -product_position_limit - selling_pos)
                selling_pos += sell_amount
                assert(sell_amount < 0)
                orders.append(Order(product, bid, sell_amount))
                self.market_taking.append((product, sell_amount, False))
		
			# if at parity, sell up until we are no longer leveraged
            if bid == acceptable_sell_price + price_aggression and selling_pos > 0:
                sell_amount = max(-vol, -selling_pos)
                selling_pos += sell_amount
                assert(sell_amount < 0)
                orders.append(Order(product, bid, sell_amount))
                self.market_taking.append((product, sell_amount, False))

		# start market making with remaining quota
		# if selling_pos
        if -product_position_limit - selling_pos < 0:
            if selling_pos > -THRESHOLDS[product]["over"]:
                target_sell_price = max(acceptable_sell_price + price_aggression, lowest_sell_price - 1)
                vol = -selling_pos - THRESHOLDS[product]["over"]
                orders.append(Order(product, target_sell_price, vol))
                selling_pos += vol
            if -THRESHOLDS[product]["over"] >= selling_pos >= -THRESHOLDS[product]["mid"]:
                target_sell_price = max(acceptable_sell_price + 1 + price_aggression, lowest_sell_price - 1)
                vol = -selling_pos - THRESHOLDS[product]["mid"]
                orders.append(Order(product, target_sell_price, vol))
                selling_pos += vol
            if -THRESHOLDS[product]["mid"] >= selling_pos:
                target_sell_price = max(acceptable_sell_price + 2 + price_aggression, lowest_sell_price - 1)
                vol = -product_position_limit - selling_pos
                orders.append(Order(product, target_sell_price, vol))
                selling_pos += vol
				
        return orders

    def black_scholes_price(self, S, K, t, r, sigma):
        def cdf(x):
            return 0.5 * (1 + math.erf(x/math.sqrt(2)))

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        price = S * cdf(d1) - K * np.exp(-r * t) * cdf(d2)
        return price

    def newtons_method(self, f, x0=0.02, epsilon=1e-7, max_iter=100, h=1e-5):
        def numerical_derivative(f, x, h=1e-5):
            return (f(x + h) - f(x - h)) / (2 * h)
        
        x = x0
        for i in range(max_iter):
            fx = f(x)
            if abs(fx) < epsilon:
                return x
            dfx = numerical_derivative(f, x, h)
            if dfx == 0:
                raise ValueError("Derivative zero. No solution found.")
            x -= fx / dfx
        raise ValueError("Maximum iterations reached. No solution found.")

    def refresh_runner_state(self, state: TradingState):
        try:
            previous_state = jsonpickle.decode(state.traderData)
            self.POSITION = previous_state.get('POSITION', {})
            self.update_starfruit_price_history(previous_state, state)
            self.update_conversions(previous_state, state)
            self.update_combined_gift_basket_price_history(previous_state, state)
            # UPDATE COCONUT
            self.COUPON_IV_STORE = previous_state.get('COUPON_IV_STORE', [])
            self.PREV_COCONUT_PRICE = previous_state.get('PREV_COCONUT_PRICE', -1)
            self.PREV_COUPON_PRICE = previous_state.get('PREV_COUPON_PRICE', -1)
            self.COUPON_IV_STORE_SIZE = previous_state.get('COUPON_IV_STORE_SIZE', 100)
        except:
            pass

    def run(self, state: TradingState):
        # state.traderData restore
        self.refresh_runner_state(state)

        # update position
        for product in state.position:
            self.POSITION[product] = state.position.get(product, 0)

        result = {"STARFRUIT": [], "AMETHYSTS": [], "ORCHIDS": [], "GIFT_BASKET": [], "CHOCOLATE": [], "STRAWBERRIES": [], "ROSES": []}
        conversions = 0

        # ORCHIDS
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orchids_orders = []
            coconut_orders = []
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
            # NEW (BASKET + ROSES + CHOCOLATE + STRAWBERRIES)
            elif product == 'GIFT_BASKET' or product == 'ROSES' or product == 'CHOCOLATE' or product == 'STRAWBERRIES':
                if product_acceptable_price is None:
                    continue
                product_acceptable_buy_price = math.floor(product_acceptable_price)
                product_acceptable_sell_price = math.ceil(product_acceptable_price)
                orders = self.get_orders_new(state, product_acceptable_sell_price, product_acceptable_buy_price, product, PRICE_AGGRESSION[product])
                result[product] = orders
            
            # COCONUT
            elif product == 'COCONUT_COUPON':
                items = ['COCONUT', 'COCONUT_COUPON']
                mid_price, best_bid_price, best_ask_price = {}, {}, {}

                for item in items:
                    _, best_sell_price = self.get_volume_and_best_price(state.order_depths[item].sell_orders, buy_order=False)
                    _, best_buy_price = self.get_volume_and_best_price(state.order_depths[item].buy_orders, buy_order=True)

                    mid_price[item] = (best_sell_price + best_buy_price) / 2
                    best_bid_price[item] = best_buy_price
                    best_ask_price[item] = best_sell_price
            
                iv = self.newtons_method(lambda sigma: self.black_scholes_price(mid_price['COCONUT'], 10_000, 250, 0, sigma) - mid_price['COCONUT_COUPON'])
                self.COUPON_IV_STORE.append(iv)

                if len(self.COUPON_IV_STORE) >= self.COUPON_IV_STORE_SIZE:
                    iv_mean, iv_std = np.mean(self.COUPON_IV_STORE), np.std(self.COUPON_IV_STORE)

                    diff = iv - iv_mean
                    INF = int(1e9)

                    if diff > self.COUPON_Z_SCORE * iv_std:
                        coconut_orders += self.calculate_orchids_orders(product, order_depth, -INF, best_bid_price['COCONUT_COUPON'])
                    elif diff < -self.COUPON_Z_SCORE * iv_std:
                        coconut_orders += self.calculate_orchids_orders(product, order_depth, best_ask_price['COCONUT_COUPON'], INF)
                    self.COUPON_IV_STORE.pop(0)
                
                self.PREV_COCONUT_PRICE = mid_price['COCONUT']
                self.PREV_COUPON_PRICE = mid_price['COCONUT_COUPON']
                result['COCONUT_COUPON'] = coconut_orders

        logger.print(f"ORCHIDS CONVERSIONS: {conversions}")

        # encode
        data_to_encode = {
            'POSITION': self.POSITION,
            'previous_starfruit_prices': self.previous_starfruit_prices,
            'previous_gift_basket_prices': self.previous_gift_basket_prices,
            'previous_chocolate_prices': self.previous_chocolate_prices,
            'previous_strawberry_prices': self.previous_strawberry_prices,
            'previous_rose_prices': self.previous_rose_prices,
            'previous_premium_basket_prices': self.previous_premium_basket_prices,
            'market_taking': self.market_taking,
            'next_market_taking': self.next_market_taking,
            'PREV_COCONUT_PRICE': self.PREV_COCONUT_PRICE,
            'PREV_COUPON_PRICE': self.PREV_COUPON_PRICE,
            'COUPON_IV_STORE': self.COUPON_IV_STORE,
            'COUPON_IV_STORE_SIZE': self.COUPON_IV_STORE_SIZE
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

        if orchids:
            penny_sell = best_sell_price - 5

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
