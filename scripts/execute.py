# ---------------------------------------------------------------------------------------------------
# By: Ruth Carvalho, Samuel Gallo-Will, Danisa Ivanovic-Zuvic, Daniel Jeyaraj, Nabiha Naweed
# Last Updated: November 17, 2026
# Purpose:
# - Execute the evolutionary algorithm.
# - Navigate to the project directory and in terminal run: `python -m scripts.execute`
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------
# Package imports.
import pandas as pd
import numpy as np
import random
import copy
from pathlib import Path

# Operations.
operations = ['+', '-', '*', '/']

# Helper function imports.
from scripts.utils import (
    print_tree,
    evaluate_tree,
    tree_size,
    initialize_population,
    tournament_selection,
    subtree_crossover,
    subtree_mutation,
    constant_mutation,
    survivor_selection
)

# Config import.
from scripts.config import GPConfig
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------
# Execution structure of the genetic programming algorithm.

def run_gp(config: GPConfig, seed: int = 42) -> tuple:
    """
    Purpose: execute the genetic programming algorithm.
    
    Args:
        config (GPConfig): Algorithm parameters.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        best_tree (list): Best expression tree.
        best_fitness (float): Lowest MSE found.
        history (pd.DataFrame): Fitness statistics by generation.
    """
    
    # Set random seed for reproducibility.
    random.seed(seed)
    
    # Load the dataset.
    data = pd.read_csv(config.data_path)
    
    # Initialize results.
    best_tree = None
    best_fitness = float('inf')
    history = []
    
    # Initialize the population.
    population = initialize_population(
        population_size = config.population_size,
        max_tree_depth = config.max_tree_depth
    )
    
    # Begin evolution.
    for generation in range(config.generations + 1):
        
        # Evaluate fitnes of population.
        fitness_scores = [
            evaluate_tree(tree, data) for tree in population
        ]
        
        # Assign infinite fitness to non-finite values.
        fitness_scores = [
            score if np.isfinite(score) else float('inf') for score in fitness_scores
        ]
        
        # Identify the best individual.
        best_index = min(
            range(len(population)), key = lambda i: fitness_scores[i]
        )
        
        current_fitness = fitness_scores[best_index]
        
        # If the current solution is better than the best so far, update.
        if best_tree is None or current_fitness < best_fitness:
            best_tree = copy.deepcopy(population[best_index])
            best_fitness = current_fitness
        
        # Record generation statistics.
        valid_scores = [
            score for score in fitness_scores if np.isfinite(score)
        ]
        mean_fitness = (
            float(np.mean(valid_scores)) if valid_scores else float('inf')
        )
        history.append({
            'generation': generation,
            'best_fitness': best_fitness,
            'mean_fitness': mean_fitness,
            'mean_tree_size': np.mean([
                tree_size(tree) for tree in population
            ])
        })
        
        # Display progress.
        if generation % 10 == 0:
            print(
                f"Generation {generation}:"
                f"Best Fitness (Lowest MSE) = {best_fitness:.4f}"
            )
        
        # Stop after the final generation.
        if generation == config.generations:
            break
        
        # Generate offspring.
        offspring = []
        
        offspring_needed = (
            config.population_size - config.elite_count
        )
        
        while len(offspring) < offspring_needed:
            
            # Parent selection.
            parent1 = tournament_selection(
                population = population,
                fitness_scores = fitness_scores,
                tournament_size = config.tournament_size
            )
            parent2 = tournament_selection(
                population = population,
                fitness_scores = fitness_scores,
                tournament_size = config.tournament_size
            )
            
            # Choose a reproductive operator.
            probability = random.random()
            
            # Recombination.
            if probability < config.crossover_probability:
                
                child1, child2 = subtree_crossover(
                    parent1 = parent1,
                    parent2 = parent2,
                    crossover_probability = 1.0,
                    max_tree_depth = config.max_tree_depth,
                    max_attempts = config.crossover_max_attempts
                )
            
            elif probability < (
                config.crossover_probability + config.mutation_probability
            ):
                
                if random.random() < 0.60:
                    child1 = subtree_mutation(
                        tree = parent1,
                        mutation_probability = 1.0,
                        max_tree_depth = config.max_tree_depth
                    )
                    child2 = subtree_mutation(
                        tree = parent2,
                        mutation_probability = 1.0,
                        max_tree_depth = config.max_tree_depth
                    )
                else:
                    child1 = constant_mutation(
                        tree = parent1,
                        mutation_probability = 1.0,
                        max_tree_depth = config.max_tree_depth
                    )
                    child2 = constant_mutation(
                        tree = parent2,
                        mutation_probability = 1.0,
                        max_tree_depth = config.max_tree_depth
                    )
            
            # Direct reproduction.
            else:
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)
                
            offspring.extend([child1, child2])
        
        # Survivor selection.
        population = survivor_selection(population, fitness_scores, offspring, config.elite_count)
    
    # Convert generation statistics to a DataFrame.
    history = pd.DataFrame(history)
    
    return best_tree, best_fitness, history

# ---------------------------------------------------------------------------------------------------
# Execute.

if __name__ == "__main__":
    
    # Instantiate config class.
    config = GPConfig()
    
    # Execute genetic programming.
    best_tree, best_fitness, history = run_gp(config)
    
    # Display results.
    print("\nRun Complete")
    print("Best expression:", print_tree(best_tree))
    print("Best MSE:", best_fitness)
    
    # Save generation statistics.
    Path("results").mkdir(exist_ok = True)
    history.to_csv("results/gp_history.csv", index = False)