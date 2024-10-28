import pandas as pd
import jsonpickle
import numpy as np
from typing import List, Dict
from datamodel import TradingState, OrderDepth, Order

class Trader:
    def __init__(self):
        self.position_limits = {"STARFRUIT": 20, "AMETHYSTS": 20}
        self.last_prices = {product: [] for product in self.position_limits}
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

    def calculate_linear_regression_and_predict(self, product: str, lambda_reg=0.1):
        if len(self.last_prices[product]) < 5:
            return None, None  # Not enough data
        
        price_data = pd.DataFrame(self.last_prices[product], columns=['mid_price'])
        for i in range(1, 5):
            price_data[f'mid_price_t-{i}'] = price_data['mid_price'].shift(i)
        price_data.dropna(inplace=True)

        X = price_data[[f'mid_price_t-{i}' for i in range(1, 5)]].values
        y = price_data['mid_price'].values
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])

        # Apply Ridge Regression by adding lambda times the identity matrix to X^TX
        theta = np.linalg.inv(X_with_intercept.T @ X_with_intercept + lambda_reg * np.eye(X_with_intercept.shape[1])) @ X_with_intercept.T @ y
        
        latest_features = np.hstack([1, X[-1, :]])
        predicted_mid_price = latest_features @ theta
        return predicted_mid_price, theta
    
    def update_market_conditions(self, state: TradingState):
        for product in self.position_limits.keys():
            if product not in state.order_depths or not self.last_prices[product]:
                continue
            
            predicted_mid_price, regression_coefficients = self.calculate_linear_regression_and_predict(product)
            if predicted_mid_price is not None and regression_coefficients is not None:
                self.market_analysis[product]['Predicted_Mid_Price'] = predicted_mid_price
                self.market_analysis[product]['Regression_Coefficients'] = regression_coefficients

            self.market_analysis[product]['SMA'] = self.calculate_dynamic_sma(product)
            self.market_analysis[product]['EMA'] = self.calculate_dynamic_ema(product)
            self.market_analysis[product]['RSI'] = self.calculate_dynamic_rsi(product)
            self.market_analysis[product]['Volatility'] = self.calculate_dynamic_volatility(product)
            
            bollinger_bands = self.calculate_dynamic_bollinger_bands(product)
            if bollinger_bands[0] is not None:
                self.market_analysis[product]['Bollinger_Upper'], self.market_analysis[product]['Bollinger_Lower'] = bollinger_bands[1], bollinger_bands[2]

    def adjust_weight_based_on_changes(self, product, current_price):
        prev_price = self.last_prices.get(product, [current_price])[-1]
        price_change = abs((current_price - prev_price) / prev_price) if prev_price else 0
        dynamic_threshold = self.calculate_dynamic_thresholds(product)

        # Be more aggressive in adjusting weights: Lower the threshold for considering a change significant.
        if price_change > (dynamic_threshold * 0.5):  # More responsive to changes
            weight_historical = 0.3  # Increase reliance on current changes
        else:
            weight_historical = 0.7

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

    def calculate_dynamic_margin_and_weight(self, order_depth: OrderDepth):
        spread = self.calculate_spread(order_depth)

        # More aggressive strategy: trade on narrower margins in highly liquid markets
        if spread <= 5:  # Very narrow spread, high liquidity
            margin = 0.01  # Very aggressive margin
            weight = 0.05  # Highly responsive
        elif spread > 5 and spread <= 20:  # Moderate conditions
            margin = 0.03
            weight = 0.15
        else:  # Wide spread, possibly lower liquidity
            margin = 0.07  # Still aggressive, but cautious
            weight = 0.25

        return margin, weight

    def calculate_spread(self, order_depth: OrderDepth):
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return float('inf')
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        spread = best_ask - best_bid
        return spread

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


    def estimate_fair_value(self, product: str, market_analysis: dict):
        """
        Focuses on the Predicted_Mid_Price for fair value, considering volatility and Bollinger Bands.
        """
        fair_value = market_analysis['Predicted_Mid_Price']

        # Check if Volatility is not None before applying adjustment
        if market_analysis['Volatility'] is not None:
            volatility_adjustment = 1 - (market_analysis['Volatility'] / 100)
            fair_value *= volatility_adjustment
        else:
            # Handle the case where Volatility is None. You might choose to not adjust fair_value
            # Or use a default volatility adjustment factor
            # Example: volatility_adjustment = 1 - (default_volatility / 100)
            pass

        # Adjust fair value based on Bollinger Band position for market sentiment
        if fair_value > market_analysis['Bollinger_Upper']:
            fair_value_adjustment = (fair_value - market_analysis['Bollinger_Upper']) / 2
            fair_value -= fair_value_adjustment
        elif fair_value < market_analysis['Bollinger_Lower']:
            fair_value_adjustment = (market_analysis['Bollinger_Lower'] - fair_value) / 2
            fair_value += fair_value_adjustment

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

        order_depth = state.order_depths[product]
        margin, weight = self.calculate_dynamic_margin_and_weight(order_depth)

        current_price = self.get_current_price(order_depth)
        acceptable_buy_price = fair_value - (fair_value * margin)
        acceptable_sell_price = fair_value + (fair_value * margin)

        if current_price:
            price_adjustment_factor = weight
            acceptable_buy_price = ((acceptable_buy_price * (1 - price_adjustment_factor)) + 
                                    (price_adjustment_factor * current_price))
            acceptable_sell_price = ((acceptable_sell_price * (1 - price_adjustment_factor)) + 
                                    (price_adjustment_factor * current_price))

        # Log for troubleshooting - providing insights into decision-making process
        print(f"{product} - Fair Value: {fair_value}, Current: {current_price}, Acceptable Buy: {acceptable_buy_price}, Acceptable Sell: {acceptable_sell_price}, Margin: {margin}, Weight: {weight}")

        # Adjusting quantity based on market conditions and trader's positions
        quantity_multiplier = 1.5  # This can be dynamically adjusted based on your analysis or confidence
        quantity_to_buy = int(min(position_limit - current_position, 10) * quantity_multiplier)
        quantity_to_sell = int(min(current_position, 5) * quantity_multiplier)

        # Generating buy order if the market price is equal or below the acceptable buy price and within position limits
        if current_position < position_limit and current_price and current_price <= acceptable_buy_price:
            orders.append(Order(product, acceptable_buy_price, quantity_to_buy))
            print(f"Placing buy order for {product} - Price: {acceptable_buy_price}, Quantity: {quantity_to_buy}")

        # Generating sell order if the market price is equal or above the acceptable sell price and we hold positions
        if current_position > 0 and current_price and current_price >= acceptable_sell_price:
            orders.append(Order(product, acceptable_sell_price, -quantity_to_sell))
            print(f"Placing sell order for {product} - Price: {acceptable_sell_price}, Quantity: {quantity_to_sell}")

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
                self.last_prices = previous_state.get('last_prices', {})
            except Exception as e:
                print(f"Error decoding traderData: {e}")

        for product, trades in state.market_trades.items():
            if trades:
                last_trade_price = trades[-1].price
                self.update_price_history(product, last_trade_price)
                print(f"Updated price history for {product}: {self.last_prices[product]}")

        self.update_market_conditions(state)

        result = {}
        conversions = []

        for product in self.position_limits.keys():
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                print(f"Market depth for {product} - Best ask: {best_ask}, Best bid: {best_bid}")

                fair_value = self.estimate_fair_value(product, self.market_analysis[product])
                print(f"Fair value for {product}: {fair_value}")
                orders = self.generate_dynamic_orders(product, fair_value, state)
                print(f"Generated orders for {product}: {orders}")
                result[product] = orders
            else:
                print(f"No order depth data for {product}")

        traderData = jsonpickle.encode({'last_prices': self.last_prices})

        return result, conversions, traderData
