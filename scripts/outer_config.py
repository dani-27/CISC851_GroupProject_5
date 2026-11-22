
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------------------------------
# Outer EA Configuration

@dataclass
class OuterConfig:

    # Outer population and initialization.
    population_size: int = 10
    generations: int = 10

    # Parent selection.
    tournament_size: int = 2

    # Recombination.
    crossover_probability: float = 0.9

    # Mutation.
    mutation_probability: float = 0.4

    # Survivor selection.
    elite_count: int = 2

    # Inner GP fitness evaluation.
    evaluation_runs: int = 3

# ---------------------------------------------------------------------------------------------------
# Inner GP Parameter Search Boundaries

GP_PARAMETER_BOUNDS = {

    # Population and initialization.
    "population_size": (100, 500),
    "generations": (250, 750),
    "max_tree_depth": (2,10),

    # Parent selection.
    "tournament_size": (2, 10),

    # Recombination.
    "crossover_probability": (0.0, 1.0),
    "crossover_max_attempts": (1, 1),

    # Mutation.
    "mutation_probability": (0.0, 1.0),

    # Survivor selection.
    "elite_count": (3, 10)
}


# ---------------------------------------------------------------------------------------------------
# Parameter Types

INTEGER_PARAMETERS = {
    "population_size",
    "generations",
    "max_tree_depth",
    "tournament_size",
    "crossover_max_attempts",
    "elite_count"
}

FLOAT_PARAMETERS = {
    "crossover_probability",
    "mutation_probability"
}
