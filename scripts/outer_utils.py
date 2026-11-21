
# ---------------------------------------------------------------------------------------------------
# By: Group Project 5
# Last Updated: November 20, 2026
# Purpose:
# - Define helper functions for the outer evolutionary algorithm.
# - Initialize and validate outer individuals.
# - Convert outer individuals into inner GP configurations.
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------
# Import Statements

import random
import numpy as np
import copy

from scripts.execute import run_gp
from scripts.config import GPConfig
from scripts.outer_config import (
    OuterConfig,
    GP_PARAMETER_BOUNDS,
    INTEGER_PARAMETERS,
    FLOAT_PARAMETERS
)


# ---------------------------------------------------------------------------------------------------
# Initialization

def initialize_outer_population(config: OuterConfig) -> list:
    """
    Purpose: initialize a population of randomly generated GP configurations.

    Args:
        config (OuterConfig): Outer EA configuration.

    Returns:
        population (list): List of dictionaries representing GP configurations.
    """

    population = []

    # Generate the initial population.
    for _ in range(config.population_size):

        individual = {}

        # Generate each GP parameter.
        for parameter, (lower, upper) in GP_PARAMETER_BOUNDS.items():

            # Generate integer parameters.
            if parameter in INTEGER_PARAMETERS:
                value = random.randint(lower, upper)

            # Generate floating-point parameters.
            elif parameter in FLOAT_PARAMETERS:
                value = random.uniform(lower, upper)

            else:
                raise ValueError(f"Unknown parameter type: {parameter}")

            individual[parameter] = value

        # Ensure the individual satisfies all constraints.
        validate_individual(individual)

        population.append(individual)

    return population


# ---------------------------------------------------------------------------------------------------
# Validation

def validate_individual(individual: dict) -> bool:
    """
    Purpose: validate an outer individual against the GP parameter boundaries
    and constraints.

    Args:
        individual (dict): GP configuration represented as a dictionary.

    Returns:
        valid (bool): True if the individual is valid.

    Raises:
        ValueError: If the individual contains invalid parameters or values.
    """

    # Check that all required parameters are present.
    if set(individual.keys()) != set(GP_PARAMETER_BOUNDS.keys()):
        raise ValueError("Individual contains missing or unexpected parameters.")

    # Check parameter boundaries and types.
    for parameter, (lower, upper) in GP_PARAMETER_BOUNDS.items():

        value = individual[parameter]

        if parameter in INTEGER_PARAMETERS:

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{parameter} must be an integer.")

        elif parameter in FLOAT_PARAMETERS:

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{parameter} must be numeric.")

        # Check minimum and maximum values.
        if not lower <= value <= upper:
            raise ValueError(
                f"{parameter} must be between {lower} and {upper}."
            )

    # Check tournament size.
    if individual["tournament_size"] > individual["population_size"]:
        raise ValueError("Tournament size cannot exceed population size.")

    # Check elite count.
    if individual["elite_count"] >= individual["population_size"]:
        raise ValueError("Elite count must be smaller than population size.")

    return True


# ---------------------------------------------------------------------------------------------------
# GP Configuration Conversion

def create_gp_config(individual: dict) -> GPConfig:
    """
    Purpose: convert an outer individual into an inner GP configuration.

    Args:
        individual (dict): GP configuration represented as a dictionary.

    Returns:
        gp_config (GPConfig): Configuration object for the inner GP.
    """

    # Validate the individual before conversion.
    validate_individual(individual)

    # Create the inner GP configuration.
    gp_config = GPConfig(**individual)

    return gp_config


# ---------------------------------------------------------------------------------------------------
# Fitness Evaluation


def evaluate_outer_individual(
    individual: dict,
    config: OuterConfig
) -> tuple:
    """
    Evaluate a GP configuration using multiple independent inner GP runs.

    Returns:
        mean_fitness: Mean best MSE across all runs.
        fitness_scores: Best MSE from each run.
        seeds: Random seeds used.
        best_tree: Tree from the run with the lowest MSE.
        best_tree_mse: MSE associated with that tree.
    """

    gp_config = create_gp_config(individual)

    if config.evaluation_runs < 1:
        raise ValueError("Evaluation runs must be at least 1.")

    fitness_scores = []
    best_trees = []

    seed_generator = random.SystemRandom()

    seeds = [
        seed_generator.randrange(2**32)
        for _ in range(config.evaluation_runs)
    ]

    # Preserve the outer EA random state.
    outer_random_state = random.getstate()

    try:
        for seed in seeds:

            best_tree, best_mse, _ = run_gp(
                config=gp_config,
                seed=seed
            )

            fitness_scores.append(float(best_mse))
            best_trees.append(best_tree)

    finally:
        random.setstate(outer_random_state)

    mean_fitness = float(np.mean(fitness_scores))

    if not np.isfinite(mean_fitness):
        mean_fitness = float('inf')

    # Identify the best individual inner GP run.
    best_run_index = min(
        range(len(fitness_scores)),
        key=lambda i: fitness_scores[i]
    )

    best_tree = best_trees[best_run_index]
    best_tree_mse = fitness_scores[best_run_index]

    return (
        mean_fitness,
        fitness_scores,
        seeds,
        best_tree,
        best_tree_mse
    )


# ---------------------------------------------------------------------------------------------------
# Parent Selection

def outer_tournament_selection(
    population: list,
    fitness_scores: list[float],
    tournament_size: int
) -> dict:
    """
    Purpose: select an outer EA parent using tournament selection.

    Args:
        population (list): Population of GP configurations.
        fitness_scores (list): Mean MSE for each configuration.
        tournament_size (int): Number of competing individuals.

    Returns:
        parent (dict): Selected GP configuration.
    """

    if len(population) != len(fitness_scores):
        raise ValueError("Population and fitness scores must have equal lengths.")

    if not 1 <= tournament_size <= len(population):
        raise ValueError("Invalid tournament size.")

    # Randomly select competing individuals.
    candidates = random.sample(
        range(len(population)),
        tournament_size
    )

    # Select the individual with the lowest mean MSE.
    winner_index = min(
        candidates,
        key=lambda i: fitness_scores[i]
    )

    # Return an independent copy of the selected parent.
    parent = copy.deepcopy(population[winner_index])

    return parent

# ---------------------------------------------------------------------------------------------------
# Recombination

def outer_uniform_crossover(
    parent1: dict,
    parent2: dict,
    crossover_probability: float
) -> tuple:
    """
    Purpose: perform uniform crossover between two GP configurations.

    Args:
        parent1 (dict): First parent.
        parent2 (dict): Second parent.
        crossover_probability (float): Probability of crossover.

    Returns:
        offspring1 (dict): First offspring.
        offspring2 (dict): Second offspring.
    """

    # Create independent copies.
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)

    # Determine whether crossover occurs.
    if random.random() >= crossover_probability:
        return offspring1, offspring2

    # Exchange parameters independently.
    for parameter in GP_PARAMETER_BOUNDS:

        # Skip parameters with fixed values.
        lower, upper = GP_PARAMETER_BOUNDS[parameter]

        if lower == upper:
            continue

        # Exchange this parameter with 50% probability.
        if random.random() < 0.5:

            offspring1[parameter] = parent2[parameter]
            offspring2[parameter] = parent1[parameter]

    # Validate offspring.
    validate_individual(offspring1)
    validate_individual(offspring2)

    return offspring1, offspring2

# ---------------------------------------------------------------------------------------------------
# Mutation

def outer_mutation(
    individual: dict,
    mutation_probability: float
) -> dict:
    """
    Purpose: mutate GP hyperparameters independently.

    Args:
        individual (dict): Parent GP configuration.
        mutation_probability (float): Per-parameter mutation probability.

    Returns:
        offspring (dict): Mutated GP configuration.
    """

    offspring = copy.deepcopy(individual)

    # Mutate each parameter independently.
    for parameter, (lower, upper) in GP_PARAMETER_BOUNDS.items():

        # Do not mutate fixed parameters.
        if lower == upper:
            continue

        # Determine whether this parameter mutates.
        if random.random() >= mutation_probability:
            continue

        # Calculate Gaussian mutation step size.
        sigma = 0.1 * (upper - lower)

        # Apply Gaussian perturbation.
        new_value = offspring[parameter] + random.gauss(0, sigma)

        # Restrict value to allowable boundaries.
        new_value = max(lower, min(upper, new_value))

        # Preserve integer parameter types.
        if parameter in INTEGER_PARAMETERS:
            new_value = int(round(new_value))

        offspring[parameter] = new_value

    # Validate the mutated individual.
    validate_individual(offspring)

    return offspring

# ---------------------------------------------------------------------------------------------------
# Survivor Selection

def outer_survivor_selection(
    population: list,
    fitness_scores: list[float],
    offspring: list,
    elite_count: int
) -> list:
    """
    Purpose: perform generational survivor selection with elitism.

    Args:
        population (list): Current outer population.
        fitness_scores (list): Mean MSE values of current individuals.
        offspring (list): Newly generated configurations.
        elite_count (int): Number of elite individuals to preserve.

    Returns:
        new_population (list): Next outer population.
    """

    if len(population) != len(fitness_scores):
        raise ValueError("Population and fitness scores must have equal lengths.")

    if not 0 <= elite_count <= len(population):
        raise ValueError("Invalid elite count.")

    # Rank individuals by fitness (lowest MSE first).
    ranked_indices = sorted(
        range(len(population)),
        key=lambda i: fitness_scores[i]
    )

    # Preserve the best individuals.
    elites = [
        copy.deepcopy(population[i])
        for i in ranked_indices[:elite_count]
    ]

    # Determine remaining population positions.
    remaining = len(population) - elite_count

    if len(offspring) < remaining:
        raise ValueError("Not enough offspring to fill the population.")

    # Fill remaining positions with offspring.
    new_population = elites + [
        copy.deepcopy(child)
        for child in offspring[:remaining]
    ]

    return new_population

from scripts.outer_config import OuterConfig

from scripts.outer_utils import (
    initialize_outer_population,
    validate_individual,
    outer_tournament_selection,
    outer_uniform_crossover,
    outer_mutation,
    outer_survivor_selection
)

# Initialize outer EA.
config = OuterConfig()

population = initialize_outer_population(config)

print("Population size:", len(population))

# Use artificial fitness scores to test the operators.
# These are not actual GP results.
fitness_scores = [float(i) for i in range(len(population))]

# Test parent selection.
parent1 = outer_tournament_selection(
    population,
    fitness_scores,
    config.tournament_size
)

parent2 = outer_tournament_selection(
    population,
    fitness_scores,
    config.tournament_size
)

# Test crossover.
child1, child2 = outer_uniform_crossover(
    parent1,
    parent2,
    config.crossover_probability
)

# Test mutation.
child1 = outer_mutation(child1, config.mutation_probability)
child2 = outer_mutation(child2, config.mutation_probability)

# Validate offspring.
assert validate_individual(child1)
assert validate_individual(child2)

# Generate enough offspring for survivor selection.
offspring = []

while len(offspring) < config.population_size - config.elite_count:
    offspring.append(outer_mutation(parent1, config.mutation_probability))

# Test survivor selection.
new_population = outer_survivor_selection(
    population,
    fitness_scores,
    offspring,
    config.elite_count
)

assert len(new_population) == config.population_size
assert all(validate_individual(ind) for ind in new_population)

# ---------------------------------------------------------------------------------------------------
# Fitness Evaluation


def evaluate_outer_individual(
    individual: dict,
    config: OuterConfig
) -> tuple:

    gp_config = create_gp_config(individual)

    if config.evaluation_runs < 1:
        raise ValueError("Evaluation runs must be at least 1.")

    fitness_scores = []
    best_trees = []

    seed_generator = random.SystemRandom()

    seeds = [
        seed_generator.randrange(2**32)
        for _ in range(config.evaluation_runs)
    ]

    outer_random_state = random.getstate()

    try:
        for seed in seeds:

            best_tree, best_mse, _ = run_gp(
                config=gp_config,
                seed=seed
            )

            fitness_scores.append(float(best_mse))
            best_trees.append(best_tree)

    finally:
        random.setstate(outer_random_state)

    mean_fitness = float(np.mean(fitness_scores))

    if not np.isfinite(mean_fitness):
        mean_fitness = float('inf')

    best_run_index = min(
        range(len(fitness_scores)),
        key=lambda i: fitness_scores[i]
    )

    best_tree = best_trees[best_run_index]
    best_tree_mse = fitness_scores[best_run_index]

    return (
        mean_fitness,
        fitness_scores,
        seeds,
        best_tree,
        best_tree_mse
    )