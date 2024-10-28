import jsonpickle
from datamodel import TradingState, OrderDepth, Order
from typing import Dict, List

class Trader:

    def __init__(self):
        # Position limits for each product to ensure we do not exceed maximum allowed positions.
        self.position_limits = {"PRODUCT1": 20, "PRODUCT2": 20}
        # Dictionary to track the last trade price for each product. This can help in estimating fair value.
        self.last_prices = {}  

    def run(self, state: TradingState):
        if state.traderData:
            try:
                previous_state = jsonpickle.decode(state.traderData)
                self.last_prices = previous_state.get('last_prices', {})
            except Exception as e:
                print(f"Error decoding traderData: {e}")

        # Main trading logic executed each iteration. Receives current market state.
        print(f"Running trader for timestamp: {state.timestamp}")
        result = {}  # Dictionary to hold orders to be placed in this iteration.
        for product, order_depth in state.order_depths.items():
            fair_value = self.estimate_fair_value(product, state)  # Estimate fair value for the product.
            # Generate buy or sell orders based on the estimated fair value and current market conditions.
            current_position = state.position.get(product, 0)
            orders = self.generate_orders(product, order_depth, current_position, fair_value)
            result[product] = orders
            print(f"Generated orders for {product}: {orders}")
        
        # Determine if any conversions should be made this iteration. Placeholder for conversion logic.
        conversions = self.determine_conversions(state)
        print(f"Conversions determined: {conversions}")
        
        # Serialize the trader's state for persistence between iterations. Includes tracking of last prices.
        traderData = jsonpickle.encode({'last_prices': self.last_prices})
        print(f"Persisting state: {traderData}")
        
        return result, conversions, traderData

    def generate_orders(self, product, order_depth, current_position, fair_value):
        # Generate buy/sell orders based on market conditions and the trader's strategy.
        orders = []
        if product == "STARFRUIT":
            acceptable_price_buy = fair_value * 0.9
            acceptable_price_sell = fair_value * 1.1
        else:
            acceptable_price_buy = fair_value * 0.95  # Set threshold for buying below fair value.
            acceptable_price_sell = fair_value * 1.05  # Set threshold for selling above fair value.

        # Using .get() to safe access
        position_limit = self.position_limits.get(product, 10)

        # Generate buy orders if sell orders are below acceptable buy price.
        for price, qty in order_depth.sell_orders.items():
            if price <= acceptable_price_buy:
                # Ensure the order does not exceed position limits.
                quantity_to_buy = min(-qty, position_limit - current_position)
                if quantity_to_buy > 0:
                    orders.append(Order(product, price, quantity_to_buy))
        # Generate sell orders if buy orders are above acceptable sell price.
        for price, qty in order_depth.buy_orders.items():
            if price >= acceptable_price_sell:
                # Ensure the order does not exceed position limits.
                quantity_to_sell = min(qty, current_position + position_limit)
                if quantity_to_sell > 0:
                    orders.append(Order(product, price, -quantity_to_sell))
        return orders

    def estimate_fair_value(self, product, state):
        # Simple model for fair value estimation. Currently uses average of last prices.
        # Improvement: Integrate a more sophisticated model using historical data, machine learning predictions,
        # and market observations to estimate a more accurate fair value.
        if product not in self.last_prices or not self.last_prices[product]:
            return 10  # Default value if no price history is available.
        return sum(self.last_prices[product]) / len(self.last_prices[product])

    def determine_conversions(self, state: TradingState):
        # Placeholder for conversion logic. Currently does not perform conversions.
        # Improvement: Logic to evaluate when conversions are profitable, considering
        # conversion costs, market conditions, and expected product value changes.
        return 0

# Example instantiation and use:
# trader = Trader()
# state = TradingState(...)  # Assuming TradingState object is appropriately populated.
# result, conversions, traderData = trader.run(state)
# Use the output (result, conversions, traderData) as needed.
