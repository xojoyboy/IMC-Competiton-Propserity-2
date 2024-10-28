import pandas as pd
import matplotlib.pyplot as plt
import statistics
import os

# Define data path and file names
base_path = "../../data/round-3-island-data-bottle/"
file_names = ["prices_round_3_day_0.csv", "prices_round_3_day_1.csv", "prices_round_3_day_2.csv"]

# Load data
df = pd.DataFrame()
for file_name in file_names:
    file_path = os.path.join(base_path, file_name)
    new_df = pd.read_csv(file_path, delimiter=";")
    df = pd.concat([df, new_df], ignore_index=True)

# Calculate the number of entries to analyze
size = len(df)

# Extract mid prices for each product from the data
basket_midprices = df[df["product"] == "GIFT_BASKET"]['mid_price'].tolist()[:size]
straw_midprices = df[df["product"] == "STRAWBERRIES"]['mid_price'].tolist()[:size]
choco_midprices = df[df["product"] == "CHOCOLATE"]['mid_price'].tolist()[:size]
roses_midprices = df[df["product"] == "ROSES"]['mid_price'].tolist()[:size]

# Calculate the difference between the price of the gift basket and the sum of its components
difference = [basket - (4 * choco + 6 * straw + rose) for basket, choco, straw, rose in zip(basket_midprices, choco_midprices, straw_midprices, roses_midprices)]

# Define the standard deviation multiplier
std_factor = 0.7

# Calculate the mean and standard deviation of the difference
mean = statistics.mean(difference)
std = statistics.stdev(difference)

# Plot the price differences
plt.figure(figsize=(20, 10))
plt.plot(difference, label='Price Difference')
plt.axhline(mean, color='r', linestyle='--', label='Mean')
plt.axhline(mean + std * std_factor, color='g', linestyle=':', label=f'Mean + {std_factor} STD')
plt.axhline(mean - std * std_factor, color='b', linestyle=':', label=f'Mean - {std_factor} STD')
plt.title('Price Difference Over Time')
plt.xlabel('Index')
plt.ylabel('Price Difference')
plt.legend()
plt.show()

print("Mean difference:", mean)
print("Standard deviation of difference:", std)
