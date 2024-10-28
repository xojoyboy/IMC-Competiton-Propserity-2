import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_and_combine_data(files):
    """
    Reads multiple CSV files and combines them into a single DataFrame.
    Args:
        files (list): A list of file paths.
    Returns:
        pd.DataFrame: Combined DataFrame from all files.
    """
    dfs = [pd.read_csv(file, delimiter=';') for file in files]  # Use delimiter=';'
    combined_df = pd.concat(dfs, ignore_index=True)
    # combined_df.sort_values(by='timestamp', inplace=True)
    return combined_df


def linear_regression_prediction(df: pd.DataFrame, product: str, num_points: int):
    """
    Performs linear regression to predict future mid prices using past data points.
    Args:
        df (pd.DataFrame): The DataFrame containing the market data.
        product (str): The product for which to perform the prediction.
        num_points (int): The number of past data points to use for the prediction.
    Returns:
        pd.DataFrame: DataFrame containing original and predicted mid prices.
        np.ndarray: The coefficients (theta) from the linear regression.
    """
    # Filter mid prices for the specified product
    mid_prices = df[df["product"] == product]["mid_price"].to_frame()
    
    # Add historical data columns based on num_points
    for i in range(1, num_points + 1):
        mid_prices[f"mid_price_t-{i}"] = mid_prices["mid_price"].shift(i)
    mid_prices.dropna(inplace=True)
    
    # Prepare features (X) and labels (y)
    X = mid_prices[[f"mid_price_t-{i}" for i in range(1, num_points + 1)]].values
    y = mid_prices["mid_price"].values
    
    # Add intercept term
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Calculate coefficients using the Normal Equation
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Predict future mid prices using theta
    mid_prices["predicted_mid_price"] = X @ theta
    
    return mid_prices, theta

# Example usage
files = [
    "../../data/round-1-island-data-bottle/prices_round_1_day_0.csv",
    "../../data/round-1-island-data-bottle/prices_round_1_day_-1.csv",
    "../../data/round-1-island-data-bottle/prices_round_1_day_-2.csv"
]
combined_df = read_and_combine_data(files)
print(combined_df.columns)

num_points = 4  # Number of historical price points to use
mid_prices, theta = linear_regression_prediction(combined_df, "STARFRUIT", num_points)

print("Coefficients (theta):", theta)
print("Last few rows of predictions:", mid_prices.tail())

# Plot mid prices and predicted mid prices
plt.plot(mid_prices.index, mid_prices["mid_price"], label="Mid Price")
plt.plot(mid_prices.index, mid_prices["predicted_mid_price"], label="Predicted Mid Price", linestyle='--')
plt.legend()
plt.show()
