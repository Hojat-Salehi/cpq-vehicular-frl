# main_dp_func.py
import numpy as np

def federated_learning_pairing(n_clients, random_seed=None):
    """
    Generate client pairing for collaborative C/R computation
    
    Args:
        n_clients (int): Number of participating clients
        random_seed (int, optional): Seed for reproducibility
    
    Returns:
        list of tuples: Client pairs, e.g., [(0, 1), (2, 3), ...]
        list of int: Seeds for each pair
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create shuffled client ordering
    ordering = list(range(n_clients))
    np.random.shuffle(ordering)
    
    pairs = []
    pair_seeds = []
    
    # Pair clients sequentially
    for i in range(0, len(ordering) - 1, 2):
        pair = (ordering[i], ordering[i + 1])
        pairs.append(pair)
        # Generate a unique seed for this pair
        pair_seed = np.random.randint(0, 1000000)
        pair_seeds.append(pair_seed)
    
    # Handle odd number of clients (last one pairs with first)
    if len(ordering) % 2 == 1:
        pair = (ordering[-1], ordering[0])
        pairs.append(pair)
        pair_seed = np.random.randint(0, 1000000)
        pair_seeds.append(pair_seed)
    
    return pairs, pair_seeds


def get_seed_for_client(client_idx, pairs, pair_seeds):
    """
    Get the pair seed for a specific client
    
    Args:
        client_idx (int): Client index
        pairs (list): List of client pairs
        pair_seeds (list): List of seeds corresponding to each pair
    
    Returns:
        int: The seed assigned to this client's pair
    """
    for pair, seed in zip(pairs, pair_seeds):
        if client_idx in pair:
            return seed
    
    # If not found (shouldn't happen), return None
    return None


# Test function
if __name__ == "__main__":
    print("="*50)
    print("Testing Pairing Function")
    print("="*50)
    
    # Test with 5 clients
    n_clients = 10
    pairs, seeds = federated_learning_pairing(n_clients, random_seed=42)
    
    print(f"\nNumber of clients: {n_clients}")
    print(f"Generated pairs: {pairs}")
    print(f"Pair seeds: {seeds}")
    
    # Verify each client gets a seed
    print("\nClient -> Seed mapping:")
    for i in range(n_clients):
        seed = get_seed_for_client(i, pairs, seeds)
        print(f"  Client {i} -> Seed {seed}")
    
    # Verify paired clients have same seed
    print("\nVerification - Paired clients have same seed:")
    for pair, seed in zip(pairs, seeds):
        client_a, client_b = pair
        seed_a = get_seed_for_client(client_a, pairs, seeds)
        seed_b = get_seed_for_client(client_b, pairs, seeds)
        print(f"  Pair {pair}: Client {client_a} seed={seed_a}, Client {client_b} seed={seed_b}, Match={seed_a == seed_b}")