import copy
import itertools

# Initialize the exchange matrix, representing exchange rates between different currencies.
EXCHANGE_MATRIX = [
    [1, 0.48, 1.52, 0.71],  # Pizza Slice
    [2.05, 1, 3.26, 1.56],  # Wasabi Root
    [0.64, 0.30, 1, 0.46],  # Snowball
    [1.41, 0.61, 2.08, 1],   # SeaShells
]

CURRENCY_NAMES = ["Pizza Slice", "Wasabi Root", "Snowball", "SeaShells"]

# Solve the maximum currency problem using dynamic programming.
def dynamic_programming_exchange():
    max_amount = [0, 0, 0, 2_000_000]  # Initial amount of currencies, only SeaShells has 2,000,000.
    best_route = [[], [], [], []]

    # Simulate five trades.
    for _ in range(5):
        new_max_amount = copy.deepcopy(max_amount)
        new_best_route = copy.deepcopy(best_route)

        for target_product in range(4):
            for origin_product in range(4):
                quantity_target = max_amount[origin_product] * EXCHANGE_MATRIX[origin_product][target_product]
                if quantity_target > new_max_amount[target_product]:
                    new_max_amount[target_product] = quantity_target
                    new_best_route[target_product] = best_route[origin_product] + [(origin_product, target_product)]

        max_amount = new_max_amount
        best_route = new_best_route

    print("Maximum currency amount:", max_amount)
    for idx, route in enumerate(best_route):
        print(f"Best route for {CURRENCY_NAMES[idx]}: ", ' -> '.join(f"{CURRENCY_NAMES[step[0]]} to {CURRENCY_NAMES[step[1]]}" for step in route))

# Use recursion to solve for maximum exchange.
def maximize_exchange_with_path(exchange_matrix, current_amount, current_currency, transaction_count, max_transactions, start_currency, path):
    if transaction_count == max_transactions:
        if current_currency != start_currency:
            return -float('inf'), []  # If the last transaction does not return to the start currency, return an invalid path.
        return current_amount, path

    best_amount = -float('inf')
    best_path = []
    for next_currency in range(len(exchange_matrix)):
        new_amount = current_amount * exchange_matrix[current_currency][next_currency]
        amount, subpath = maximize_exchange_with_path(
            exchange_matrix, new_amount, next_currency, transaction_count + 1, max_transactions, start_currency, path + [next_currency]
        )
        if amount > best_amount:
            best_amount = amount
            best_path = subpath

    return best_amount, best_path

# Use itertools to generate all possible transaction paths and find the best path.
def maximize_exchange_with_itertools(exchange_matrix, start_amount, start_currency, max_transactions):
    best_amount = -float('inf')
    best_path = []
    all_paths = itertools.product(range(len(exchange_matrix)), repeat=max_transactions)

    for path in all_paths:
        current_amount = start_amount
        current_currency = start_currency
        valid_path = True
        transaction_path = [start_currency]

        for next_currency in path:
            if current_currency == next_currency:
                valid_path = False
                break
            new_amount = current_amount * exchange_matrix[current_currency][next_currency]
            current_currency = next_currency
            current_amount = new_amount
            transaction_path.append(current_currency)

        if valid_path and current_currency == start_currency and current_amount > best_amount:
            best_amount = current_amount
            best_path = transaction_path

    return best_amount, best_path

if __name__ == "__main__":
    print("### Dynamic Programming Approach ###")
    dynamic_programming_exchange()

    print("\n### Recursive Method ###")
    start_amount = 2_000_000
    max_end_amount, transaction_path = maximize_exchange_with_path(
        EXCHANGE_MATRIX, start_amount, 3, 0, 5, 3, [3]
    )
    print("Maximum Ending Amount:", max_end_amount)
    print("Transaction Path:", ' -> '.join(CURRENCY_NAMES[i] for i in transaction_path))

    print("\n### Using itertools ###")
    max_end_amount, transaction_path = maximize_exchange_with_itertools(
        EXCHANGE_MATRIX, start_amount, 3, 5
    )
    print("Maximum Ending Amount:", max_end_amount)
    print("Transaction Path:", ' -> '.join(CURRENCY_NAMES[i] for i in transaction_path))
