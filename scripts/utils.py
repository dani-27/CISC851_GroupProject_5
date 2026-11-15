# ---------------------------------------------------------------------------------------------------
# By: Ruth Carvalho, Samuel Gallo-Will, Danisa Ivanovic-Zuvic, Daniel Jeyaraj, Nabiha Naweed
# Last Updated: November 14, 2026
# Purpose: define helper functions.

# Statement on AI
# GhatGPT-5.5 was used to assist with syntax and structure.
# All ideas are original and written by the authors.
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------
# Import Statements
import pandas as pd
import numpy as np
import random
import copy

# ---------------------------------------------------------------------------------------------------
# Global Variables

operations = ['+', '-', '*', '/']

sample_tree = [
    '/',
    ['*', 2.0, 'S'],
    ['+', 3.0, 'S']
]

# ---------------------------------------------------------------------------------------------------
# Print Trees

def print_tree(tree: list) -> str:
    """
    Purpose: convert an expression tree to a string.
    
    Args:
        tree (list): A nested list representing an equation.
    
    Outputs:
        tree_string (str): The equation in string form.
    """
    
    # Identify when recursion reaches a terminal node.
    if not isinstance(tree, (list, tuple)):
        return str(tree)

    # Internal node (operator with two children).
    operator, left, right = tree

    if operator not in operations:
        raise ValueError(f"Invalid operator: {operator}")

    left_expr = print_tree(left)
    right_expr = print_tree(right)
    
    tree_string = f"({left_expr} {operator} {right_expr})"

    return tree_string

# ---------------------------------------------------------------------------------------------------
# Evaluation Functions

def evaluate_expression(tree: list, S: float) -> float:
    """
    Purpose: evaluate the expression represented by the tree.
    
    Args:
        tree (list): A nested list representing an equation.
        S (float): Placeholder for substrate concentration.
    
    Outputs:
        result (float): The evaluated result of the expression.
    """
    
    # Identify when recursion reaches a terminal node.
    if not isinstance(tree, list):
        return S if tree == 'S' else tree

    operator, left, right = tree

    a = evaluate_expression(left, S)
    b = evaluate_expression(right, S)

    if operator == '+':
        result = a + b
    elif operator == '-':
        result = a - b
    elif operator == '*':
        result = a * b
    elif operator == '/':
        if abs(b) > 1e-10:
            result = a / b
        else:
            result = 1.0
    
    return result

def evaluate_tree(tree: list) -> float:
    """
    Purpose: calculate MSE between actual and predicted reaction rates (based on expression tree).
    
    Args:
        tree (list): A nested list representing an equation.

    Outputs:
        mse (float): Mean squared error.
    """
    
    # Import kinetics data.
    data = pd.read_csv("data/Puromycin.csv")

    # Extract substrate concentrations.
    concentrations = data.iloc[:, 0].to_numpy()
    # Extract reaction velocities.
    actual_rates = data.iloc[:, 1].to_numpy()

    # Evaluate the expression tree for each substrate concentration.
    predicted_rates = [
        evaluate_expression(tree, S) for S in concentrations
    ]

    # Calculate the mean squared error between actual and predicted rates.
    mse = float(np.mean((actual_rates - predicted_rates) ** 2))

    return mse

# ---------------------------------------------------------------------------------------------------
# Initialization Functions

def random_terminal() -> str | float:
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

def generate_tree(current_depth: int, method: str, max_tree_depth: int = 6) -> list | str | float:
    """
    Purpose: generate a random expression tree using Full or Grow.
    
    Args:
        current_depth (int): Current depth of the node being generated.
        max_tree_depth (int): Maximum permitted tree depth.
        method (str): 'full' or 'grow' method for tree generation.

    Outputs:
        tree (list | str | float): A randomly generated expression tree.
    """

    # Terminal node at maximum depth.
    if current_depth == max_tree_depth:
        return random_terminal()

    # Grow method: allow branches to terminate early.
    if method == 'grow' and current_depth > 1:
        if random.random() < 0.5:
            return random_terminal()

    # Generate an operator and its two children.
    operator = random.choice(operations)

    left = generate_tree(current_depth + 1, method, max_tree_depth)
    right = generate_tree(current_depth + 1, method, max_tree_depth)

    return [operator, left, right]

def initialize_population(population_size: int, max_tree_depth: int) -> list:
    """
    Purpose: initialize a population using ramped half-and-half.

    Args:
        population_size (int): Number of trees to generate.
        max_tree_depth (int): Maximum permitted initial tree depth.

    Returns:
        population (list): List of randomly generated expression trees.
    """

    # Prevent invalid population size.
    if population_size < 1:
        raise ValueError("Population size must be at least 1.")
    
    # Prevent invalid tree depth.
    if max_tree_depth < 2:
        raise ValueError("Maximum tree depth must be at least 2.")

    population = []

    # Generate trees using ramped half-and-half.
    for i in range(population_size):

        # Alternate between Full and Grow.
        method = 'full' if i % 2 == 0 else 'grow'

        # Cycle through depths from 2 to max_tree_depth.
        depth = 2 + (i // 2) % (max_tree_depth - 1)

        tree = generate_tree(current_depth = 1, method = method, max_tree_depth = depth)

        population.append(tree)

    random.shuffle(population)

    return population

# ---------------------------------------------------------------------------------------------------
# Recombination Functions

def get_node_paths(tree: list, path: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    """
    Purpose: recursively traverse a tree and return a list of all paths from the root to each node.
    
    Args:
        tree (list): A nested list representing an equation.
        path (tuple): A tuple of integers describing how to reach the current node from the root. Default is an empty tuple (root node).
    
    Outputs:
        paths (list): A list of tuples, each tuple representing a path to a node in the tree.
    """
    
    paths = [path]

    # 1 indicates left child, 2 indicates right child.
    if isinstance(tree, list):
        paths += get_node_paths(tree[1], path + (1,))
        paths += get_node_paths(tree[2], path + (2,))

    return paths

def tree_depth(tree: list) -> int:
    """
    Purpose: calculate the maximum depth of an expression tree.

    Args:
        tree (list): A nested list representing an equation.

    Outputs:
        depth (int): The maximum depth of the tree.
    """

    # Terminal node.
    if not isinstance(tree, list):
        depth = 1

    # Internal node.
    else:
        left_depth = tree_depth(tree[1])
        right_depth = tree_depth(tree[2])

        depth = 1 + max(left_depth, right_depth)

    return depth

def subtree_crossover(
    parent1: list,
    parent2: list,
    crossover_probability: float,
    max_tree_depth: int = 6,
    max_attempts: int = 10
) -> tuple:
    """
    Purpose: perform depth-aware subtree crossover between two parent trees.

    Args:
        parent1 (list): First parent expression tree.
        parent2 (list): Second parent expression tree.
        crossover_probability (float): Probability of crossover.
        max_tree_depth (int): Maximum permitted offspring depth.
        max_attempts (int): Maximum number of crossover attempts.

    Outputs:
        tuple: Two offspring expression trees.
    """

    # Create independent copies of the parents.
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)

    # Determine whether crossover occurs.
    if random.random() >= crossover_probability:
        return offspring1, offspring2

    # Identify possible crossover points (excluding the root).
    paths1 = get_node_paths(parent1)[1:]
    paths2 = get_node_paths(parent2)[1:]

    # Retry crossover until valid offspring are produced.
    for _ in range(max_attempts):

        # Select random crossover points.
        path1 = random.choice(paths1)
        path2 = random.choice(paths2)

        # Retrieve selected subtrees.
        subtree1 = parent1
        for index in path1:
            subtree1 = subtree1[index]

        subtree2 = parent2
        for index in path2:
            subtree2 = subtree2[index]

        # Check whether crossover would exceed maximum depth.
        if (len(path1) + tree_depth(subtree2) > max_tree_depth or len(path2) + tree_depth(subtree1) > max_tree_depth):
            continue

        # Create fresh copies for the valid crossover.
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)

        # Replace selected subtree in offspring 1.
        node = offspring1
        for index in path1[:-1]:
            node = node[index]
        node[path1[-1]] = copy.deepcopy(subtree2)

        # Replace selected subtree in offspring 2.
        node = offspring2
        for index in path2[:-1]:
            node = node[index]
        node[path2[-1]] = copy.deepcopy(subtree1)

        return offspring1, offspring2

    # Return unchanged parents if all crossover attempts fail.
    return copy.deepcopy(parent1), copy.deepcopy(parent2)

# ---------------------------------------------------------------------------------------------------
# Testing

def main():
    """
    Purpose: test the helper functions.
    """

    # Test print_tree and evaluate_expression.
    test_result = evaluate_expression(tree = sample_tree, S = 1)
    print(print_tree(tree = sample_tree))
    print(test_result)
    
    # Test population initialization.
    population = initialize_population(
        population_size = 300,
        max_tree_depth = 6
    )
    # Display the first five individuals.
    for i, tree in enumerate(population[:5]):
        print(f"Individual {i + 1}: {print_tree(tree)}")

if __name__ == "__main__":
    main()