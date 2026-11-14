# ---------------------------------------------------------------------------------------------------
# By: Ruth Carvalho, Samuel Gallo-Will, Danisa Ivanovic-Zuvic, Daniel Jeyaraj, Nabiha Naweed
# Last Updated: November 13, 2026
# Purpose: helper functions to initialize a population of candidate solution trees.

# Statement on AI
# GhatGPT-5.5 was used to assist with syntax and structure.
# All ideas are original and written by the authors.
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------
# Import Statements
import random

# ---------------------------------------------------------------------------------------------------

operations = ['+', '-', '*', '/']

def random_terminal():
    """
    Purpose: generate a random terminal node.
    - 50% chance of returning 'S'.
    - 25% chance of returning low-magnitude float.
    - 25% chance of returning high-magnitude float.
    """
    if random.random() < 0.5:
        return 'S'

    if random.random() < 0.5:
        return round(random.uniform(-1, 1), 3)
    else:
        return round(random.uniform(-250, 250), 3)

def generate_tree(current_depth, max_depth, method):
    """
    Generate a random expression tree using Full or Grow.
    """

    # Terminal node at maximum depth.
    if current_depth == max_depth:
        return random_terminal()

    # Grow method: allow branches to terminate early.
    if method == 'grow' and current_depth > 1:
        if random.random() < 0.5:
            return random_terminal()

    # Generate an operator and its two children.
    operator = random.choice(operations)

    left = generate_tree(current_depth + 1, max_depth, method)
    right = generate_tree(current_depth + 1, max_depth, method)

    return [operator, left, right]


def initialize_population(population_size: int, max_tree_depth: int) -> list:
    """
    Initialize a population using Ramped Half-and-Half.

    Parameters:
        population_size (int): Number of trees to generate.
        max_tree_depth (int): Maximum permitted initial tree depth.

    Returns:
        population (list): List of randomly generated expression trees.
    """

    if max_tree_depth < 2:
        raise ValueError("Maximum tree depth must be at least 2.")

    population = []

    for i in range(population_size):

        # Alternate between Full and Grow.
        method = 'full' if i % 2 == 0 else 'grow'

        # Cycle through depths from 2 to max_tree_depth.
        depth = 2 + (i // 2) % (max_tree_depth - 1)

        tree = generate_tree(1, depth, method)

        population.append(tree)

    random.shuffle(population)

    return population