import csv
import matplotlib.pyplot as plt

# Define the path to your CSV file
csv_file_path = '/Users/haowu/Desktop/code/projects/IMC/Prosperity-2/data/round-1-results/round1_results_data.csv'
# Initialize dictionaries to hold the profit for each product and timestamps
profits = {
    "AMETHYSTS": [],
    "STARFRUIT": []
}
timestamps = []

# Initialize a list to hold the combined profit
combined_profit = []

# Process the CSV file
with open(csv_file_path, mode='r') as csvfile:
    # Assuming your CSV delimiter is a semicolon ';'
    csvreader = csv.DictReader(csvfile, delimiter=';')
    for row in csvreader:
        product = row['product']
        pl = float(row['profit_and_loss'])  # Convert profit/loss to float
        timestamp = float(row['timestamp'])
        
        # Append timestamp
        if timestamp not in timestamps:
            timestamps.append(timestamp)
        
        # Append profit/loss to the respective product list
        if product in profits:
            profits[product].append(pl)
        else:
            profits[product] = [pl]

# Calculate combined profit at each timestamp
for timestamp in timestamps:
    combined_pl = sum(profits[product][index] for product in profits for index, ts in enumerate(timestamps) if ts == timestamp)
    combined_profit.append(combined_pl)

# Plotting
plt.figure(figsize=(12, 7))

# Plot profit for each product
for product, values in profits.items():
    plt.plot(timestamps, values, label=product)

# Plot combined profit
plt.plot(timestamps, combined_profit, label="Combined", linestyle='--', color='black')

plt.title("Profit for Each Product and Combined")
plt.xlabel("Timestamp")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()