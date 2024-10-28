import pandas as pd
import numpy as np

class Utils:
    def __init__(self):
        self.prices_df = pd.DataFrame()
        self.market_analysis = {}  # To store analysis results

    def read_data(self, prices_file: str):
        self.prices_df = pd.read_csv(prices_file, delimiter=";")
        self.prices_df['ORCHIDS'] = pd.to_numeric(self.prices_df['ORCHIDS'], errors='coerce')

    def calculate_bollinger_bands(self, window: int = 20):
        price_data = self.prices_df['ORCHIDS']
        sma = price_data.rolling(window=window).mean()
        std_dev = price_data.rolling(window=window).std()
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        return upper_band.iloc[-1], lower_band.iloc[-1]

    def calculate_volatility(self, window: int = 20):
        price_data = self.prices_df['ORCHIDS']
        volatility = price_data.rolling(window=window).std()
        return volatility.iloc[-1]

    def analyze_market(self):
        """
        Analyzes the market for ORCHIDS and compiles various
        statistics and predictions into a single data structure, 
        organizing it within self.market_analysis.
        """
        product = 'ORCHIDS'
        self.market_analysis[product] = {}

        upper_band, lower_band = self.calculate_bollinger_bands()
        volatility = self.calculate_volatility()

        self.market_analysis[product]['Bollinger_Upper'] = upper_band
        self.market_analysis[product]['Bollinger_Lower'] = lower_band
        self.market_analysis[product]['Volatility'] = volatility

        print(f"Market Analysis for {product}:")
        for key, value in self.market_analysis[product].items():
            print(f"{key}: {value}")



def main():
    utils = Utils()
    utils.read_data("prices_orchids.csv")
    utils.analyze_market()

if __name__ == "__main__":
    main()
