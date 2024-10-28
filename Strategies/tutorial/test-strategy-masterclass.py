import pandas as pd
import jsonpickle
from typing import List, Dict

class Trader:
    def __init__(self):
        self.position_limits = {"STARFRUIT": 20, "AMETHYSTS": 20}
        self.last_prices = {}
        self.positions = {"STARFRUIT": 0, "AMETHYSTS": 0}
        self.prices_df = pd.DataFrame()
        self.trades_df = pd.DataFrame()

    def read_data(self, prices_file: str, trades_file: str):
        self.prices_df = pd.read_csv(prices_file)
        self.trades_df = pd.read_csv(trades_file)

    def estimate_fair_value_sma(self, product: str, window: int = 5):
        """Estimates the fair value of a product using Simple Moving Average (SMA) over the specified window."""
        price_data = self.observation.get_prices(product)
        sma = price_data['mid_price'].rolling(window=window).mean().iloc[-1]
        return sma

    def estimate_fair_value_ema(self, product: str, span: int = 5):
        """Estimates the fair value of a product using Exponential Moving Average (EMA) over the specified span."""
        price_data = self.observation.get_prices(product)
        ema = price_data['mid_price'].ewm(span=span, adjust=False).mean().iloc[-1]
        return ema

    def calculate_rsi(self, product: str, window: int = 14):
        """Calculates the Relative Strength Index (RSI) for the given product over the specified window."""
        price_data = self.observation.get_prices(product)
        delta = price_data['mid_price'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calculate_bollinger_bands(self, product: str, window: int = 20):
        """Calculates Bollinger Bands for the given product over the specified window."""
        price_data = self.observation.get_prices(product)
        sma = price_data['mid_price'].rolling(window=window).mean()
        std_dev = price_data['mid_price'].rolling(window=window).std()
        
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        
        # Assuming the latest data point is the current value
        current_sma = sma.iloc[-1]
        current_upper_band = upper_band.iloc[-1]
        current_lower_band = lower_band.iloc[-1]
        
        print(f"Bollinger Bands for {product} - SMA: {current_sma}, Upper Band: {current_upper_band}, Lower Band: {current_lower_band}")
        return current_sma, current_upper_band, current_lower_band

    def calculate_volatility(self, product: str, window: int = 20):
    """Calculates the volatility (standard deviation) of the mid_price for the specified product over the given window."""
    price_data = self.observation.get_prices(product)
    volatility = price_data['mid_price'].rolling(window=window).std().iloc[-1]
    print(f"Volatility for {product} over the last {window} periods: {volatility}")
    return volatility

    def analyze_high_volume_trades(self, product: str, volume_threshold: int):
    """Analyzes trades for the specified product to find trades with volumes higher than the given threshold."""
    trade_data = self.observation.get_trades(product)
    high_volume_trades = trade_data[trade_data['quantity'] > volume_threshold]
    if not high_volume_trades.empty:
        print(f"Found high volume trades for {product}:")
        print(high_volume_trades)
    else:
        print(f"No high volume trades found for {product} above the threshold of {volume_threshold}.")
    return high_volume_trades

    def analyze_market(self, product: str):
    # Initialize a dictionary to store analysis results
    market_analysis = {}
    
    # Calculate SMA and add to the analysis results
    market_analysis['SMA'] = self.estimate_fair_value_sma(product)
    
    # Calculate EMA and add to the analysis results
    market_analysis['EMA'] = self.estimate_fair_value_ema(product)
    
    # Calculate RSI and add to the analysis results
    market_analysis['RSI'] = self.calculate_rsi(product)
    
    # Calculate Bollinger Bands and add to the analysis results
    bollinger_bands = self.calculate_bollinger_bands(product)
    market_analysis['Bollinger_Upper'], market_analysis['Bollinger_Lower'] = bollinger_bands
    
    # Calculate Volatility and add to the analysis results
    market_analysis['Volatility'] = self.calculate_volatility(product)
    
    # Analyze high volume trades and add to the analysis results
    # Assuming a placeholder threshold for high volume trades
    high_volume_threshold = 100  # Placeholder threshold, adjust based on your market analysis
    high_volume_trades = self.analyze_high_volume_trades(product, high_volume_threshold)
    market_analysis['High_Volume_Trades'] = high_volume_trades
    
    # Print or return the market analysis results
    print(f"Market Analysis for {product}:")
    for key, value in market_analysis.items():
        print(f"{key}: {value}")
    
    return market_analysis

    def estimate_fair_value_starfruit(self):
        # Implement logic based on self.prices_df to estimate STARFRUIT's fair value
        return self.prices_df[self.prices_df['product'] == 'STARFRUIT']['mid_price'].mean()

    def estimate_fair_value_amethysts(self):
        # Implement logic based on self.prices_df to estimate AMETHYSTS's fair value
        return self.prices_df[self.prices_df['product'] == 'AMETHYSTS']['mid_price'].mean()

    def generate_orders_starfruit(self, fair_value):
        orders = []
        if fair_value < 5000:  # Example condition
            orders.append({"product": "STARFRUIT", "action": "BUY", "quantity": 10, "price": fair_value})
        return orders

    def generate_orders_amethysts(self, fair_value):
        orders = []
        if fair_value > 10000:  # Example condition
            orders.append({"product": "AMETHYSTS", "action": "SELL", "quantity": 5, "price": fair_value})
        return orders

    def determine_conversions(self):
        # Implement any conversion strategy here, possibly using self.positions
        conversions = []
        if self.positions["STARFRUIT"] > 10:
            conversions.append({"from": "STARFRUIT", "to": "AMETHYSTS", "quantity": 10})
        return conversions

    def run(self, prices_file: str, trades_file: str):
        print("Reading data...")
        self.read_data(prices_file, trades_file)

        print("Estimating fair values...")
        fair_value_starfruit = self.estimate_fair_value_starfruit()
        fair_value_amethysts = self.estimate_fair_value_amethysts()

        print("Generating orders...")
        orders_starfruit = self.generate_orders_starfruit(fair_value_starfruit)
        orders_amethysts = self.generate_orders_amethysts(fair_value_amethysts)

        print("Determining conversions...")
        conversions = self.determine_conversions()

        print("Orders for STARFRUIT:", orders_starfruit)
        print("Orders for AMETHYSTS:", orders_amethysts)
        print("Conversions:", conversions)

        # Combine orders and conversions for final decision
        final_decisions = orders_starfruit + orders_amethysts + conversions
        return final_decisions

# Example usage
trader = Trader()
final_decisions = trader.run('prices.csv', 'trades.csv')
print("Final Decisions:", jsonpickle.encode(final_decisions, unpicklable=False))
