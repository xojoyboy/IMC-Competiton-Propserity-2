def find_paths(current_node, steps_remaining, current_product, path, all_paths, adj_matrix, max_product):
    # If no steps left, check if we are back at 'Shells' and update the path with the highest product
    if steps_remaining == 0:
        if current_node == 3:  # 'Shells' is represented by index 3
            if current_product > max_product[0]:
                max_product[0] = current_product
                all_paths.clear()  # Found a new path with a higher product
                all_paths.append(path.copy())
            elif current_product == max_product[0]:
                all_paths.append(path.copy())
        return

    # Try moving to all other nodes based on the adjacency matrix
    for next_node in range(4):
        if current_node != next_node and adj_matrix[current_node][next_node] > 0:  # Valid and non-zero edge
            # Move to the next node
            path.append(next_node)
            find_paths(next_node, steps_remaining - 1, current_product * adj_matrix[current_node][next_node],
                       path, all_paths, adj_matrix, max_product)
            path.pop()  # Backtrack

def find_maximum_product_path(adj_matrix, start_node):
    all_paths = []
    max_product = [0]  # Store the product of the path with the highest product
    find_paths(start_node, 5, 1, [start_node], all_paths, adj_matrix, max_product)
    return all_paths, max_product[0]

# Define the adjacency matrix based on the provided weights
adj_matrix = [
    # pizza, Wasabi Root, Snowball, Shells
    [1, 0.48, 1.52, 0.71],   # from pizza
    [2.05, 1, 3.26, 1.56],   # from Wasabi Root
    [0.64, 0.3, 1, 0.46],    # from Snowball
    [1.41, 0.61, 2.08, 1]    # from Shells
]

# Start from 'Shells' (index 3)
start_node = 3

# Run the algorithm to find the path with the maximum product of weights
valid_paths, max_product = find_maximum_product_path(adj_matrix, start_node)

# Convert indices back to names for display
names = ['pizza', 'Wasabi Root', 'Snowball', 'Shells']
named_paths = [[names[node] for node in path] for path in valid_paths]

print(f"Maximum Path Product: {max_product}")
for i, path in enumerate(named_paths, 1):
    print(f"Path {i}: {' -> '.join(path)}")
