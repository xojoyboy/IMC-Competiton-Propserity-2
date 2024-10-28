import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


def func4(X, a, b, c, d, e):
    return a*X[:, 0] + b*X[:, 1] + c*X[:, 2] + d*X[:, 3] + e

def func6(X, a, b, c, d, e, f, g):
    return a*X[:, 0] + b*X[:, 1] + c*X[:, 2] + d*X[:, 3] + e*X[:, 4] + f*X[:, 5] +g

def func8(X, a, b, c, d, e, f, g, h, i):
    return (a*X[:, 0] + b*X[:, 1] + c*X[:, 2] + d*X[:, 3] + e*X[:, 4] + f*X[:, 5] +g*X[:, 6]
+ h*X[:, 7] + i)


def perform_curve_fit_and_plot(df: pd.DataFrame, product: str, num_points: int, func: callable):
    # Filter data for the specified product
    product_data = df[df["product"] == product]
    best_bid_price = product_data["bid_price_1"]
    best_bid_amount = product_data["bid_volume_1"]
    
    best_ask_price = product_data["ask_price_1"]
    best_ask_amount = product_data["ask_volume_1"]
    
    # Calculate average_best_price
    average_best_price = (best_bid_price * best_ask_amount + best_ask_price * best_bid_amount) / (best_ask_amount + best_bid_amount)
    
    # Convert to DataFrame for easier manipulation
    average_best_price_df = average_best_price.to_frame(name='average_best_price')
    
    # Add historical data columns based on num_points
    for i in range(1, num_points+1):
        average_best_price_df[f"average_best_price_t-{i}"] = average_best_price_df["average_best_price"].shift(i)
    
    average_best_price_df.dropna(inplace=True)  # Drop rows with NaN values
    
    # Prepare X and y for curve fitting
    X = average_best_price_df[[f"average_best_price_t-{i}" for i in range(1, num_points + 1)]].values
    y = average_best_price_df["average_best_price"].values
    
    # Perform curve fitting
    popt, pcov = curve_fit(func, X, y)
    print("Optimal parameters:", popt)
    
    # Predict using the fitted model
    predicted_average_best_price = func(X, *popt)
    average_best_price_df["predicted_average_best_price"] = predicted_average_best_price
    
        # Calculating metrics
    mae = mean_absolute_error(y, predicted_average_best_price)
    mse = mean_squared_error(y, predicted_average_best_price)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y, predicted_average_best_price)
    
    # Printing out metrics
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r_squared:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(average_best_price_df['average_best_price'], label='Actual Average Best Price', marker='o')
    plt.plot(average_best_price_df['predicted_average_best_price'], label='Predicted Average Best Price', linestyle='--', marker='x')
    plt.title('Actual vs Predicted Average Best Prices')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return popt


# Example usage
files = [
    "../../round-1-island-data-bottle/prices_round_1_day_0.csv",
    "../../round-1-island-data-bottle/prices_round_1_day_-1.csv",
    "../../round-1-island-data-bottle/prices_round_1_day_-2.csv"
]
combined_df = read_and_combine_data(files)

num_points = 6  # Number of historical price points to use
mid_prices, popt = perform_curve_fit_and_plot(combined_df, "STARFRUIT", num_points, func6)
