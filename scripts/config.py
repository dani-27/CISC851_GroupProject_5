# ---------------------------------------------------------------------------------------------------
# By: Ruth Carvalho, Samuel Gallo-Will, Danisa Ivanovic-Zuvic, Daniel Jeyaraj, Nabiha Naweed
# Last Updated: November 17, 2026
# Purpose: define parameter configuration for a run.
# ---------------------------------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class GPConfig:
    # Data path.
    data_path: str = "data/Puromycin.csv"
    
    # Population and initialization.
    population_size: int = 300
    generations: int = 500
    max_tree_depth: int = 4
    
    # Parent selection.
    tournament_size: int = 3
    
    # Recombination.
    crossover_probability: float = 0.7
    crossover_max_attempts: int = 10
    
    # Mutation.
    mutation_probability: float = 0.25
    
    # Survivor selection.
    elite_count: int = 3