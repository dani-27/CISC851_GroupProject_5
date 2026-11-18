# ---------------------------------------------------------------------------------------------------
# By: Ruth Carvalho, Samuel Gallo-Will, Danisa Ivanovic-Zuvic, Daniel Jeyaraj, Nabiha Naweed
# Last Updated: November 16, 2026
# Purpose: define helper functions.
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------
# Import Statements
import pandas as pd
import numpy as np
import random
import copy
from pathlib import Path
from dataclasses import dataclass
from scripts.config import GPConfig

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
# Evaluation

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
        return S if tree == 'S' else float(tree)

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

def evaluate_tree(tree: list, data: pd.DataFrame) -> float:
    """
    Purpose: calculate MSE between actual and predicted reaction rates (based on expression tree).
    
    Args:
        tree (list): A nested list representing an equation.
        data (pd.DataFrame): substrate concentrations and actual reaction rates.

    Outputs:
        mse (float): Mean squared error.
    """
    
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

    # Handle cases where MSE is not finite.
    if not np.isfinite(mse):
        mse = float('inf')
    
    return mse

def tree_size(tree: list) -> int:
    """
    Purpose: calculate the number of nodes in a tree.
    
    Args:
        tree (list): A nested list representing an equation.

    Returns:
        num_nodes (int): Total number of nodes in a tree.
    """
    
    if not isinstance(tree, list):
        return 1
    
    num_nodes = 1 + tree_size(tree[1]) + tree_size(tree[2])
    
    return num_nodes

# ---------------------------------------------------------------------------------------------------
# Initialization

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
# Parent Selection

def tournament_selection(
    population: list,
    fitness_scores: list[float],
    tournament_size: int = 3
) -> list:
    """
    Purpose: select a parent using tournament selection.

    Args:
        population (list): Population of expression trees.
        fitness_scores (list): Fitness of each individual.
        tournament_size (int): Number of competing individuals.

    Outputs:
        parent (list): Individual with the lowest MSE.
    """

    # Randomly select competing individuals.
    candidates = random.sample(
        range(len(population)), tournament_size
    )

    # Identify the individual with the lowest MSE.
    winner_index = min(
        candidates,
        key=lambda i: fitness_scores[i]
    )

    parent = population[winner_index]

    return parent

# ---------------------------------------------------------------------------------------------------
# Recombination

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

def test_subtree_crossover():
    """
    Purpose: test the correctness of depth-aware subtree crossover.
    """

    random.seed(42)

    parent1 = ['/', 'S', ['+', 'S', 3.0]]
    parent2 = ['/', ['*', 200.0, 'S'], ['+', 'S', 0.1]]

    original1 = copy.deepcopy(parent1)
    original2 = copy.deepcopy(parent2)

    # Test 1: No crossover when probability is zero.
    child1, child2 = subtree_crossover(
        parent1, parent2, crossover_probability=0.0
    )

    assert child1 == parent1
    assert child2 == parent2

    print("Test 1 passed: Zero crossover probability.")

    # Test 2: Perform multiple crossover attempts.
    successful_changes = 0

    for i in range(100):
        child1, child2 = subtree_crossover(
            parent1,
            parent2,
            crossover_probability=1.0,
            max_tree_depth=6,
            max_attempts=10
        )

        # Verify that original parents remain unchanged.
        assert parent1 == original1
        assert parent2 == original2

        # Verify that offspring remain expression trees.
        assert isinstance(child1, list)
        assert isinstance(child2, list)

        # Verify that tree depths do not exceed the limit.
        assert tree_depth(child1) <= 6
        assert tree_depth(child2) <= 6

        # Count crossover events that introduce variation.
        if child1 != parent1 or child2 != parent2:
            successful_changes += 1

        # Display the first five crossover attempts.
        if i < 5:
            print(f"\nAttempt {i + 1}")
            print("Offspring 1:", print_tree(child1))
            print("Offspring 2:", print_tree(child2))

    print("\nTest 2 passed: Offspring satisfy constraints.")
    print(f"Variation introduced: {successful_changes}/100")

    # Test 3: Verify crossover introduces variation.
    assert successful_changes > 0

    print("Test 3 passed: Crossover introduces variation.")
    print("\nAll tests passed!")

# ---------------------------------------------------------------------------------------------------
# Mutation Functions

def subtree_mutation(
    tree: list,
    mutation_probability: float,
    max_tree_depth: int = 6
) -> list:
    """
    Purpose: perform depth-aware subtree mutation on an expression tree.

    Args:
        tree (list): Parent expression tree.
        mutation_probability (float): Probability of mutation.
        max_tree_depth (int): Maximum permitted tree depth.

    Outputs:
        offspring (list): Mutated expression tree.
    """

    # Create an independent copy of the parent.
    offspring = copy.deepcopy(tree)

    # Determine whether mutation occurs.
    if random.random() >= mutation_probability:
        return offspring

    # Select a random mutation point.
    path = random.choice(get_node_paths(offspring))

    # Determine the maximum depth available for the new subtree.
    available_depth = max_tree_depth - len(path)

    # Select the depth of the replacement subtree.
    min_depth = 2 if not path else 1
    new_depth = random.randint(min_depth, available_depth)

    # Generate a random replacement subtree.
    if new_depth == 1:
        new_subtree = random_terminal()
    else:
        method = random.choice(['full', 'grow'])
        new_subtree = generate_tree(1, method, new_depth)

    # Replace the entire tree if the root is selected.
    if not path:
        offspring = new_subtree

    # Otherwise, replace the selected subtree.
    else:
        node = offspring
        for index in path[:-1]:
            node = node[index]

        node[path[-1]] = new_subtree

    return offspring

def constant_mutation(
    tree: list,
    mutation_probability: float,
    max_tree_depth: int = 6
) -> list:
    """
    Purpose: mutate a numerical constant within an expression tree.
    
    Args:
        tree (list): Parent expression tree.
        mutation_probability (float): Probability of mutation.
        max_tree_depth (int): Maximum permitted offspring depth.

    Returns:
        offspring ():
    """
    offspring = copy.deepcopy(tree)

    # Determine whether mutation occurs.
    if random.random() >= mutation_probability:
        return offspring

    # Find all numerical constant terminals.
    constant_paths = []

    for path in get_node_paths(offspring):
        node = offspring

        for index in path:
            node = node[index]

        if isinstance(node, (int, float)):
            constant_paths.append(path)

    # Fall back to subtree mutation if no constants exist.
    if not constant_paths:
        return subtree_mutation(offspring, 1.0, max_tree_depth)

    # Select a numerical constant.
    path = random.choice(constant_paths)

    node = offspring
    for index in path[:-1]:
        node = node[index]

    # Apply Gaussian perturbation.
    old_value = node[path[-1]]
    sigma = max(0.01, 0.1 * abs(old_value))

    node[path[-1]] = round(
        old_value + random.gauss(0, sigma), 5
    )

    return offspring

def test_subtree_mutation():
    """
    Purpose: test the correctness of depth-aware subtree mutation.
    """
    
    parent = ['/', ['*', 200.0, 'S'], ['+', 'S', 0.1]]
    original = copy.deepcopy(parent)

    # Test mutation probability of zero.
    child = subtree_mutation(parent, 0.0, 6)
    assert child == parent

    # Test repeated mutations.
    changes = 0

    for _ in range(100):
        child = subtree_mutation(parent, 1.0, 6)

        assert tree_depth(child) <= 6
        assert parent == original
        assert isinstance(child, list)

        if child != parent:
            changes += 1

    print("All tests passed!")
    print(f"Mutations producing changes: {changes}/100")

# ---------------------------------------------------------------------------------------------------
# Survivor Selection

def survivor_selection(
    population: list,
    fitness_scores: list[float],
    offspring: list,
    elite_count: int = 6
) -> list:
    """
    Purpose: perform generational survivor selection with elitism.

    Args:
        population (list): Current population.
        fitness_scores (list): Fitness of current individuals.
        offspring (list): Newly generated individuals.
        elitism (int): Number of elite individuals retained.

    Outputs:
        new_population (list): Population for the next generation.
    """

    # Rank individuals by fitness.
    ranked_indices = sorted(
        range(len(population)),
        key=lambda i: fitness_scores[i]
    )

    # Preserve the best individuals.
    elites = [
        copy.deepcopy(population[i])
        for i in ranked_indices[:elite_count]
    ]

    # Fill the remaining positions with offspring.
    remaining = len(population) - elite_count

    new_population = elites + offspring[:remaining]

    return new_population

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
    
    # Test subtree crossover.
    test_subtree_crossover()

    # Test subtree mutation.
    test_subtree_mutation()

if __name__ == "__main__":
    main()