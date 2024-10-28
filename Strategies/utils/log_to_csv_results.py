import csv
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the log and output CSV files
log_file_path = os.path.join(script_dir, "../../data/round-1-results/round_1.log")
output_csv_path = os.path.join(script_dir, "../../data/round-1-results/round1_results_data.csv")

# Define a function to process the log file and extract relevant data
def process_log_file(log_file_path, output_csv_path):
    with open(log_file_path, 'r') as log_file, open(output_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';')
        # Write the header row to the CSV file
        csv_writer.writerow(['day', 'timestamp', 'product', 'bid_price_1', 'bid_volume_1', 'bid_price_2', 'bid_volume_2', 'bid_price_3', 'bid_volume_3', 'ask_price_1', 'ask_volume_1', 'ask_price_2', 'ask_volume_2', 'ask_price_3', 'ask_volume_3', 'mid_price', 'profit_and_loss'])
        
        # Initialize a flag to track whether we are within the relevant section
        in_relevant_section = False
        
        skip_lines = 0
        # Process each line in the log file
        for line in log_file:
            # Check if we have reached the start of the relevant section
            if 'Activities log:' in line:
                in_relevant_section = True
                skip_lines += 1
                continue  # Skip the line with "Activities log:"
            
            if skip_lines == 1:
                skip_lines += 1
                continue

            if line.strip().startswith('{'):
                in_relevant_section = False
                
            # Check if we have reached the end of the relevant section
            if '1;999900;' in line:
                break  # Exit the loop once we hit the timestamp of 999900
            
            # Process lines only within the relevant section
            if in_relevant_section:
                # Split the line by semicolon to extract fields
                fields = line.strip().split(';')
                # Write the extracted fields to the CSV file
                csv_writer.writerow(fields)

# Call the function to process the log file
process_log_file(log_file_path, output_csv_path)
print("Data extraction complete. Output saved to:", output_csv_path)
