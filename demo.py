# ---------------------------------------------------------------------------------------------------
# By: Ruth Carvalho, Samuel Gallo-Will, Danisa Ivanovic-Zuvic, Daniel Jeyaraj, Nabiha Naweed
# Last Updated: November 13, 2026
# Purpose: [INSERT PURPOSE HERE LATER]

# Statement on AI
# GhatGPT-5.5 was used to assist with syntax and structure.
# All ideas are original and written by the authors.
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------
# Import Statements
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------------------------------


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
        result = a / b if abs(b) > 1e-10 else 1.0
    
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