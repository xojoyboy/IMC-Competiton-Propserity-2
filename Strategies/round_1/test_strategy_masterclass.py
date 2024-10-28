import pandas as pd
import jsonpickle
import numpy as np
from typing import List, Dict
from datamodel import TradingState, OrderDepth, Order

class Trader:
    def __init__(self):
        self.position_limits = {"STARFRUIT": 20, "AMETHYSTS": 20}
        self.last_prices = {}
        self.last_volumes = {}
        self.change_thresholds = {}
        self.positions = {"STARFRUIT": 0, "AMETHYSTS": 0}
        self.market_analysis = {
            "AMETHYSTS": {
                'SMA': 9999.7,
                'EMA': 9999.498935442854,
                'RSI': 45.945945945945944,
                'Bollinger_Upper': 10001.9208400562,
                'Bollinger_Lower': 9997.979159943801,
                'Volatility': 0.985420028099787,
                'High_Volume_Trades': pd.DataFrame(columns=['timestamp', 'buyer', 'seller', 'symbol', 'currency', 'price', 'quantity']),
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
                'High_Volume_Trades': pd.DataFrame(columns=['timestamp', 'buyer', 'seller', 'symbol', 'currency', 'price', 'quantity']),
                'Predicted_Mid_Price': 5052.8361918519295,
                'Regression_Coefficients': np.array([51.51121501, 0.87492925, -0.49278925, 0.52432386, 0.08366426])
            }
        }

    def update_market_conditions(self, state: TradingState):
        """
        update_market_conditions: Update the market conditions based on the latest order depth data.
        params: state (TradingState) - The current state of the market.
        """  
        for product, order_depth in state.order_depths.items():
            if product not in self.market_analysis:
                continue  # Skip products not in initial analysis

            current_price = self.get_current_price(order_depth)

            new_sma = self.calculate_dynamic_sma(product)
            new_ema = self.calculate_dynamic_ema(product)
            new_rsi = self.calculate_dynamic_rsi(product)
            new_volatility = self.calculate_dynamic_volatility(product)
            new_bollinger_bands = self.calculate_dynamic_bollinger_bands(product)

            # No need to calculate new_mid_price if using price history for calculations
            # If needed for other calculations, uncomment the following line
            # new_mid_price = self.get_current_price(product, order_depth)
            
            # Update market_analysis with new indicators
            self.adjust_indicators_based_on_changes(product, current_price, new_sma, new_ema, new_rsi, new_volatility, new_bollinger_bands)

    def adjust_indicators_based_on_changes(self, product, current_price, new_sma, new_ema, new_rsi, new_volatility, new_bollinger_bands):
        """Adjusts the market indicators based on significant changes."""

        if current_price is None:
            return
        
        weight_historical, weight_current = self.adjust_weight_based_on_changes(product, current_price)

        if new_sma is not None:
            self.market_analysis[product]['SMA'] = (self.market_analysis[product].get('SMA', new_sma) * weight_historical) + (new_sma * weight_current)
        if new_ema is not None:
            self.market_analysis[product]['EMA'] = (self.market_analysis[product].get('EMA', new_ema) * weight_historical) + (new_ema * weight_current)
        if new_rsi is not None:
            self.market_analysis[product]['RSI'] = (self.market_analysis[product].get('RSI', new_rsi) * weight_historical) + (new_rsi * weight_current)
        if new_volatility is not None:
            self.market_analysis[product]['Volatility'] = (self.market_analysis[product].get('Volatility', new_volatility) * weight_historical) + (new_volatility * weight_current)

        if new_bollinger_bands[0] is not None and new_bollinger_bands[1] is not None and new_bollinger_bands[2] is not None:
            upper_band_weighted = (self.market_analysis[product].get('Bollinger_Upper', new_bollinger_bands[1]) * weight_historical) + (new_bollinger_bands[1] * weight_current)
            lower_band_weighted = (self.market_analysis[product].get('Bollinger_Lower', new_bollinger_bands[2]) * weight_historical) + (new_bollinger_bands[2] * weight_current)
            self.market_analysis[product]['Bollinger_Upper'] = upper_band_weighted
            self.market_analysis[product]['Bollinger_Lower'] = lower_band_weighted


    def adjust_weight_based_on_changes(self, product, current_price):
        """
        Dynamically adjusts weights based on the change in price, considering historical changes.
        """
        prev_price = self.last_prices.get(product, [current_price])[-1]  # Get last or current if not available
        price_change = abs((current_price - prev_price) / prev_price) if prev_price else 0

        # Define initial thresholds for price changes
        initial_threshold = 0.01  # Adjust based on historical analysis

        # Dynamically adjust threshold based on the product's historical volatility
        dynamic_threshold = self.calculate_dynamic_thresholds(product)

        # Adjust weight for historical data based on the significance of current change
        if price_change > dynamic_threshold:
            weight_historical = 0.4  # Less weight on historical data if current change is significant
        else:
            weight_historical = 0.6  # More weight on historical data if current change is within expectations

        # Ensure weights are within bounds [0, 1] and calculate current data weight
        weight_historical = min(1, max(0, weight_historical))
        weight_current = 1 - weight_historical

        return weight_historical, weight_current

    def calculate_dynamic_thresholds(self, product):
        """
        Calculates dynamic thresholds for changes in price based on historical data.
        Uses volatility (standard deviation of price changes) as a metric.
        """
        if product not in self.last_prices or len(self.last_prices[product]) < 2:
            return 0.01  # Default threshold if insufficient data

        prices = self.last_prices[product]
        price_changes = [abs((prices[i] - prices[i-1]) / prices[i-1]) for i in range(1, len(prices))]
        volatility = np.std(price_changes)

        # Adjust threshold based on volatility
        dynamic_threshold = min(max(2 * volatility, 0.01), 0.05)  # Ensuring reasonable bounds
        return dynamic_threshold

    def calculate_dynamic_volatility(self, product: str, window: int = 20):
        if product not in self.last_prices or len(self.last_prices[product]) < window:
            return None
        recent_prices = self.last_prices[product][-window:]
        sma = sum(recent_prices) / len(recent_prices)
        std_dev = (sum([(p - sma) ** 2 for p in recent_prices]) / len(recent_prices)) ** 0.5
        return std_dev
        
    def calculate_dynamic_bollinger_bands(self, product: str, window: int = 20):
        if product not in self.last_prices or len(self.last_prices[product]) < window:
            return None, None, None
        recent_prices = self.last_prices[product][-window:]
        sma = sum(recent_prices) / len(recent_prices)
        std_dev = (sum([(p - sma) ** 2 for p in recent_prices]) / len(recent_prices)) ** 0.5
        upper_band = sma + 2 * std_dev
        lower_band = sma - 2 * std_dev
        return sma, upper_band, lower_band
    def calculate_dynamic_rsi(self, product: str, window: int = 14):
        if product not in self.last_prices or len(self.last_prices[product]) < window + 1:
            return None
        gains = []
        losses = []
        for i in range(1, window + 1):
            delta = self.last_prices[product][-i] - self.last_prices[product][-i - 1]
            if delta > 0:
                gains.append(delta)
            else:
                losses.append(abs(delta))
        average_gain = sum(gains) / window
        average_loss = sum(losses) / window
        rs = average_gain / average_loss if average_loss else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def calculate_dynamic_ema(self, product: str, span: int = 5, alpha: float = None):
        if product not in self.last_prices or len(self.last_prices[product]) < span:
            return None
        if alpha is None:
            alpha = 2 / (span + 1)
        prices = self.last_prices[product][-span:]
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * alpha + ema
        return ema

    def calculate_dynamic_sma(self, product: str, window: int = 5):
        if product in self.last_prices and len(self.last_prices[product]) >= window:
            recent_prices = self.last_prices[product][-window:]
            sma = sum(recent_prices) / len(recent_prices)
            return sma
        return None

    def update_price_history(self, product: str, last_trade_price: float):
        """
        Updates the price history for a given product with the latest trade price.

        params: product (str) - The product symbol.
                last_trade_price (float) - The price of the last trade.
        """
        if product not in self.last_prices:
            self.last_prices[product] = []
        self.last_prices[product].append(last_trade_price)

        # Optional: Limit the size of the price history to maintain only recent data
        max_history_size = 100  # Example: keep last 100 price points
        self.last_prices[product] = self.last_prices[product][-max_history_size:]

    def update_volume_history(self, product: str, last_trade_volume: int):
        """
        Updates the volume history for a given product with the latest trade volume.
    
        params: product (str) - The product symbol.
                last_trade_volume (int) - The volume of the last trade.
        """
        if product not in self.last_volumes:
            self.last_volumes[product] = []
        self.last_volumes[product].append(last_trade_volume)
    
        # Optional: Limit the size of the volume history to maintain only recent data
        max_history_size = 100  # Example: keep last 100 volume points
        self.last_volumes[product] = self.last_volumes[product][-max_history_size:]

    def estimate_fair_value(self, product: str, market_analysis: dict):
        """
        Estimates the fair value of a product based on comprehensive market analysis, considering
        multiple indicators, volatility, high-volume trades, and the position relative to Bollinger Bands.
        """
        # Starting with the predicted mid price as a base
        fair_value = market_analysis['Predicted_Mid_Price']

        # Combine SMA and EMA for trend analysis
        trend_indicators = [market_analysis['SMA'], market_analysis['EMA']]
        trend_average = sum(trend_indicators) / len(trend_indicators)

        # Incorporate Bollinger Bands for volatility and price position analysis
        if 'Bollinger_Upper' in market_analysis and 'Bollinger_Lower' in market_analysis:
            bollinger_position = None
            if fair_value > market_analysis['Bollinger_Upper']:
                bollinger_position = "above_upper"
            elif fair_value < market_analysis['Bollinger_Lower']:
                bollinger_position = "below_lower"
            else:
                bollinger_position = "within_bands"
            
            # Adjust fair value based on Bollinger Band position
            if bollinger_position == "above_upper":
                # Price might be too high, adjust downwards
                fair_value_adjustment = (fair_value - market_analysis['Bollinger_Upper']) / 2
                fair_value -= fair_value_adjustment
            elif bollinger_position == "below_lower":
                # Price might be too low, adjust upwards
                fair_value_adjustment = (market_analysis['Bollinger_Lower'] - fair_value) / 2
                fair_value += fair_value_adjustment
        
        # Consider market volatility - higher volatility might warrant a more conservative estimate
        volatility_adjustment = 1 - (market_analysis['Volatility'] / 100)
        fair_value *= volatility_adjustment
        
        # High-volume trades can indicate strong market interest - adjust fair value accordingly
        if not market_analysis['High_Volume_Trades'].empty:
            # Example: increase fair value by a percentage based on the volume
            high_volume_adjustment = market_analysis['High_Volume_Trades']['quantity'].sum() / 10000
            fair_value += fair_value * high_volume_adjustment

        # Final check to ensure fair value is between SMA and EMA for consistency
        fair_value = min(max(fair_value, trend_average), max(trend_indicators))

        return fair_value

    def get_current_price(self, order_depth: OrderDepth):
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None  # Can't calculate if either side is empty
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        current_price = (best_ask + best_bid) / 2
        return current_price

    def analyze_order_depth(self, order_depth):
        """分析给定的订单深度数据，并计算平均买价和平均卖价。"""
        def weighted_average_price(orders):
            total_volume = sum(orders.values())
            if total_volume == 0:
                return None
            weighted_sum = sum(price * volume for price, volume in orders.items())
            return weighted_sum / total_volume

        avg_buy_price = weighted_average_price(order_depth.buy_orders)
        avg_sell_price = weighted_average_price(order_depth.sell_orders)

        return avg_buy_price, avg_sell_price


    def generate_dynamic_orders(self, product, fair_value, state: TradingState):
        orders = []
        position_limit = self.position_limits[product]
        current_position = state.position.get(product, 0)
        print("current_position", current_position)

        # Retrieve the current order depth for the given product
        order_depth = state.order_depths[product]
        avg_buy_price, avg_sell_price = self.analyze_order_depth(order_depth)
        
        # Calculate the current price as the midpoint between the best buy and sell prices
        current_price = self.get_current_price(order_depth)
        print("current_price ", current_price)
        print("fair_value", fair_value)

        margin = fair_value * 0.05  # Example margin percentage to define acceptable price range around fair value
        print("margin", margin)
        acceptable_buy_price = fair_value - margin
        acceptable_sell_price = fair_value + margin

        print("acceptable_buy_price ", acceptable_buy_price)
        print("acceptable_sell_price ", acceptable_sell_price)

        # Incorporate the current market price to adjust the acceptable buy and sell prices further
        if current_price:
            price_adjustment_factor = 0.1  # Example adjustment factor for current price influence
            acceptable_buy_price = (acceptable_buy_price + price_adjustment_factor * current_price) / (1 + price_adjustment_factor)
            acceptable_sell_price = (acceptable_sell_price + price_adjustment_factor * current_price) / (1 + price_adjustment_factor)

        # Enhance buy and sell prices based on order depth analysis
        if avg_buy_price is not None and avg_sell_price is not None:
            adjustment_weight = 0.2  # Example adjustment factor for order depth influence
            acceptable_buy_price = ((1 - adjustment_weight) * acceptable_buy_price) + (adjustment_weight * avg_buy_price)
            acceptable_sell_price = ((1 - adjustment_weight) * acceptable_sell_price) + (adjustment_weight * avg_sell_price)

        # Generate buy order if below fair value and position limit not reached
        if current_position < position_limit and avg_sell_price <= acceptable_buy_price:
            quantity_to_buy = min(position_limit - current_position, 10)  # Example max buy quantity
            orders.append(Order(product, avg_sell_price, quantity_to_buy))

        # Generate sell order if above fair value and holding positions
        if current_position > 0 and avg_buy_price >= acceptable_sell_price:
            quantity_to_sell = min(current_position, 5)  # Example max sell quantity
            orders.append(Order(product, avg_buy_price, -quantity_to_sell))

        return orders

    def determine_conversions(self):
        # Implement any conversion strategy here, possibly using self.positions
        conversions = []
        if self.positions["STARFRUIT"] > 10:
            conversions.append({"from": "STARFRUIT", "to": "AMETHYSTS", "quantity": 10})
        return conversions

    def run(self, state: TradingState):
        if state.traderData:
            try:
                previous_state = jsonpickle.decode(state.traderData)
                # Ensure both price and volume histories are loaded
                self.last_prices = previous_state.get('last_prices', {})
                self.last_volumes = previous_state.get('last_volumes', {})
            except Exception as e:
                print(f"Error decoding traderData: {e}")

        # Update both price and volume histories based on market trades
        for product, trades in state.market_trades.items():
            if trades:
                last_trade_price = trades[-1].price
                last_trade_volume = trades[-1].quantity  # Assuming 'quantity' is the attribute for trade volume
                self.update_price_history(product, last_trade_price)
                self.update_volume_history(product, last_trade_volume)

        # Update market conditions before generating orders
        self.update_market_conditions(state)

        # Dict to store orders for each product
        result = {}
        conversions = 0  # Placeholder for conversion strategy

        # Analyze market and generate orders for each product
        for product in self.position_limits.keys():
            if product in state.order_depths:
                # No need to pass state to analyze_market since market_analysis is already updated
                fair_value = self.estimate_fair_value(product, self.market_analysis[product])
                orders = self.generate_dynamic_orders(product, fair_value, state)
                print("orders", orders)
                result[product] = orders

        # Serialize the trader's state, including both price and volume histories, for persistence
        traderData = jsonpickle.encode({'last_prices': self.last_prices, 'last_volumes': self.last_volumes})

        return result, conversions, traderData
