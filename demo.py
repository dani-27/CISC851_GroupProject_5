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
import random
import copy
# ---------------------------------------------------------------------------------------------------


operations = ['+', '-', '*', '/']

sample_tree = [
    '/',
    ['*', 2.0, 'S'],
    ['+', 3.0, 'S']
]

def print_tree(tree: list) -> str:
    
    """
    Purpose: print trees.
    
    Parameters:
        tree (list): a nested list representing an equation.
    
    Outputs:
        tree_string (str): the tree equation in string form.
    """
    
    # Terminal node (variable or constant).
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