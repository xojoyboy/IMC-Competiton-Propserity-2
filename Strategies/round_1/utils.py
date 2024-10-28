import pandas as pd
import numpy as np

class Utils:
    def __init__(self): 
        self.prices_df = pd.DataFrame()
        self.trades_df = pd.DataFrame()  
        self.market_analysis = {}  # To store analysis results
        self.change_thresholds = {}  # To store average change thresholds  

    def read_data(self, prices_file: str, trades_file: str):
                self.prices_df = pd.read_csv(prices_file, delimiter=";")
                self.trades_df = pd.read_csv(trades_file, delimiter=";")

    def estimate_fair_value_sma(self, product: str, window: int = 5):
        price_data = self.prices_df[self.prices_df['product'] == product]
        sma = price_data['mid_price'].rolling(window=window).mean().iloc[-1]
        return sma

    def estimate_fair_value_ema(self, product: str, span: int = 5):
        price_data = self.prices_df[self.prices_df['product'] == product]
        ema = price_data['mid_price'].ewm(span=span, adjust=False).mean().iloc[-1]
        return ema

    def calculate_rsi(self, product: str, window: int = 14):
        price_data = self.prices_df[self.prices_df['product'] == product]
        delta = price_data['mid_price'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calculate_bollinger_bands(self, product: str, window: int = 20):
        price_data = self.prices_df[self.prices_df['product'] == product]
        sma = price_data['mid_price'].rolling(window=window).mean()
        std_dev = price_data['mid_price'].rolling(window=window).std()
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        return upper_band.iloc[-1], lower_band.iloc[-1]


    def calculate_volatility(self, product: str, window: int = 20):
        price_data = self.prices_df[self.prices_df['product'] == product]
        volatility = price_data['mid_price'].rolling(window=window).std().iloc[-1]
        return volatility

    def analyze_high_volume_trades(self, product: str, volume_threshold: int):
        trade_data = self.trades_df[self.trades_df['symbol'] == product]
        high_volume_trades = trade_data[trade_data['quantity'] > volume_threshold]
        return high_volume_trades
    
    def calculate_linear_regression_and_predict(self, product: str):
        """
        Calculates linear regression coefficients (theta) for the specified product
        and uses them to predict the next mid price.
        """
        # Filter the price data for the given product
        price_data = self.prices_df[self.prices_df['product'] == product].copy()

        # Create shifted columns for previous prices to use as features
        for i in range(1, 5):
            price_data[f'mid_price_t-{i}'] = price_data['mid_price'].shift(i)
        price_data.dropna(inplace=True)  # Drop rows with NaN values resulting from the shift operation

        # Prepare features (X) and labels (y)
        X = price_data[[f'mid_price_t-{i}' for i in range(1, 5)]].values
        y = price_data['mid_price'].values

        # Add a column of ones to X for the intercept term
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Compute theta using the Normal Equation
        theta = np.linalg.inv(X.T @ X) @ X.T @ y

        # Use theta to predict the next mid price
        # Applying the coefficients to the most recent set of features
        latest_features = X[-1, :]  # Use the latest available features for prediction
        predicted_mid_price = latest_features @ theta

        return predicted_mid_price, theta
    
    def calculate_average_changes(self):
        """
        Calculates average percentage changes for mid_price and volume from historical CSV data.
        Stores the results in self.change_thresholds.
        """
        products = self.prices_df['product'].unique()
        for product in products:
            product_data = self.prices_df[self.prices_df['product'] == product]
            
            # Calculate percentage changes for mid_price
            mid_price_changes = product_data['mid_price'].pct_change().abs()
            avg_mid_price_change = mid_price_changes.mean()
            
            # Calculate percentage changes for volume (using a sample column here, adjust as needed)
            # Assuming bid_volume_1 as a sample for trade volume changes
            volume_changes = product_data['bid_volume_1'].pct_change().abs()
            avg_volume_change = volume_changes.mean()

            # Store average changes in a dictionary
            self.change_thresholds[product] = {
                'avg_mid_price_change': avg_mid_price_change,
                'avg_volume_change': avg_volume_change
            }

    def analyze_market(self, product: str):
        """
        Analyzes the market for a specific product and compiles various
        statistics and predictions into a single data structure, 
        organizing it within self.market_analysis.
        """
        if product not in self.market_analysis:
            self.market_analysis[product] = {}

        # Assuming other methods have been updated to dynamically calculate based on current data
        self.market_analysis[product]['SMA'] = self.estimate_fair_value_sma(product)
        self.market_analysis[product]['EMA'] = self.estimate_fair_value_ema(product)
        self.market_analysis[product]['RSI'] = self.calculate_rsi(product)
        bollinger_bands = self.calculate_bollinger_bands(product)
        if bollinger_bands:
            self.market_analysis[product]['Bollinger_Upper'], self.market_analysis[product]['Bollinger_Lower'] = bollinger_bands
        self.market_analysis[product]['Volatility'] = self.calculate_volatility(product)
        self.market_analysis[product]['High_Volume_Trades'] = self.analyze_high_volume_trades(product, 100)  # This might need an update for dynamic analysis
        predicted_mid_price, theta = self.calculate_linear_regression_and_predict(product)  # Adjust this method if necessary for dynamic data
        self.market_analysis[product]['Predicted_Mid_Price'] = predicted_mid_price
        self.market_analysis[product]['Regression_Coefficients'] = theta

        print(f"Market Analysis for {product}:")
        for key, value in self.market_analysis[product].items():
            print(f"{key}: {value}")
    
def main():
    utils = Utils()
    utils.read_data("prices.csv", "trades.csv")
    utils.calculate_average_changes()
    for product in utils.prices_df['product'].unique():
        utils.analyze_market(product)
    print(utils.change_thresholds)
    print(utils.market_analysis)


if __name__ == "__main__":
    main()
