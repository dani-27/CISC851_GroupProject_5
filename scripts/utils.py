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

# ---------------------------------------------------------------------------------------------------
# Helper Functions

operations = ['+', '-', '*', '/']

sample_tree = [
    '/',
    ['*', 2.0, 'S'],
    ['+', 3.0, 'S']
]

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

# ---------------------------------------------------------------------------------------------------
# Testing

test_result = evaluate_expression(tree = sample_tree, S = 1)
print(print_tree(tree = sample_tree))
print(test_result)