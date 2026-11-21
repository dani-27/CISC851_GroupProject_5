
import random
import copy
import pandas as pd

from pathlib import Path

from scripts.outer_config import OuterConfig

from scripts.outer_utils import (
    initialize_outer_population,
    evaluate_outer_individual,
    outer_tournament_selection,
    outer_uniform_crossover,
    outer_mutation,
    outer_survivor_selection
)

from scripts.utils import print_tree


def save_results(
    config,
    best_individual,
    best_fitness,
    best_tree,
    best_tree_mse,
    best_scores,
    best_seeds,
    history
):
    """
    Save the results of one outer EA experiment to a text file.
    """

    # Create results directory if it does not already exist.
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find the next available configuration number.
    existing_files = results_dir.glob("Config_*.txt")

    existing_numbers = []

    for file in existing_files:
        try:
            number = int(file.stem.split("_")[1])
            existing_numbers.append(number)
        except (ValueError, IndexError):
            continue

    config_number = max(existing_numbers, default=0) + 1

    output_file = results_dir / f"Config_{config_number}.txt"

    # Convert the best tree into a readable equation.
    best_equation = print_tree(best_tree)

    with open(output_file, "w", encoding="utf-8") as file:

        file.write("=" * 60 + "\n")
        file.write(f"OUTER EA EXPERIMENT: CONFIGURATION {config_number}\n")
        file.write("=" * 60 + "\n\n")

        # Outer EA settings.
        file.write("OUTER EA SETTINGS\n")
        file.write("-" * 60 + "\n")

        for parameter, value in vars(config).items():
            file.write(f"{parameter}: {value}\n")

        # Best GP configuration.
        file.write("\nBEST GP CONFIGURATION\n")
        file.write("-" * 60 + "\n")

        for parameter, value in best_individual.items():
            file.write(f"{parameter}: {value}\n")

        # Fitness results.
        file.write("\nFITNESS RESULTS\n")
        file.write("-" * 60 + "\n")

        file.write(f"Mean fitness: {best_fitness:.6f}\n")
        file.write(f"Best individual run MSE: {best_tree_mse:.6f}\n")

        file.write("\nIndividual run scores:\n")

        for i, score in enumerate(best_scores):
            file.write(f"Run {i + 1}: {score:.6f}\n")

        file.write("\nRandom seeds:\n")

        for i, seed in enumerate(best_seeds):
            file.write(f"Run {i + 1}: {seed}\n")

        # Best evolved equation.
        file.write("\nBEST EVOLVED EQUATION\n")
        file.write("-" * 60 + "\n")

        file.write(f"Equation: {best_equation}\n")
        file.write(f"Equation MSE: {best_tree_mse:.6f}\n")

        # Fitness history.
        file.write("\nOUTER EA FITNESS HISTORY\n")
        file.write("-" * 60 + "\n")

        for _, row in history.iterrows():
            file.write(
                f"Generation {int(row['generation'])}: "
                f"{row['best_fitness']:.6f}\n"
            )

    print(f"\nResults saved to: {output_file}")

    return output_file


def run_outer_ea(config: OuterConfig, seed: int = 42) -> tuple:

    random.seed(seed)

    # --------------------------------------------------
    # 1. Initialize population
    # --------------------------------------------------

    population = initialize_outer_population(config)

    print("Initial population created.")
    print(f"Population size: {len(population)}")

    # --------------------------------------------------
    # 2. Evaluate initial population
    # --------------------------------------------------

    fitness_scores = []
    best_trees = []
    best_tree_mses = []
    evaluation_scores = []
    evaluation_seeds = []

    for i, individual in enumerate(population):

        fitness, scores, seeds, tree, tree_mse = (
            evaluate_outer_individual(individual, config)
        )

        fitness_scores.append(fitness)
        best_trees.append(tree)
        best_tree_mses.append(tree_mse)
        evaluation_scores.append(scores)
        evaluation_seeds.append(seeds)

        print(
            f"Initial Individual {i + 1}/{len(population)} "
            f"- Mean MSE: {fitness:.4f}"
        )

    # Identify the best initial individual.
    best_index = min(
        range(len(fitness_scores)),
        key=lambda i: fitness_scores[i]
    )

    best_individual = copy.deepcopy(population[best_index])
    best_fitness = fitness_scores[best_index]

    best_tree = copy.deepcopy(best_trees[best_index])
    best_tree_mse = best_tree_mses[best_index]
    best_scores = evaluation_scores[best_index]
    best_seeds = evaluation_seeds[best_index]

    history = [{
        "generation": 0,
        "best_fitness": best_fitness
    }]

    print(f"\nBest initial fitness: {best_fitness:.4f}")

    # --------------------------------------------------
    # 3. Evolutionary loop
    # --------------------------------------------------

    for generation in range(1, config.generations + 1):

        print(f"\n========== Generation {generation} ==========")

        offspring_count = config.population_size - config.elite_count

        offspring = []

        # --------------------------------------------------
        # 3a. Parent selection, crossover and mutation
        # --------------------------------------------------

        while len(offspring) < offspring_count:

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

            child1, child2 = outer_uniform_crossover(
                parent1,
                parent2,
                config.crossover_probability
            )

            child1 = outer_mutation(
                child1,
                config.mutation_probability
            )

            child2 = outer_mutation(
                child2,
                config.mutation_probability
            )

            offspring.append(child1)

            if len(offspring) < offspring_count:
                offspring.append(child2)

        # --------------------------------------------------
        # 3b. Evaluate offspring
        # --------------------------------------------------

        offspring_fitness = []
        offspring_trees = []
        offspring_tree_mses = []
        offspring_scores = []
        offspring_seeds = []

        for i, individual in enumerate(offspring):

            fitness, scores, seeds, tree, tree_mse = (
                evaluate_outer_individual(individual, config)
            )

            offspring_fitness.append(fitness)
            offspring_trees.append(tree)
            offspring_tree_mses.append(tree_mse)
            offspring_scores.append(scores)
            offspring_seeds.append(seeds)

            print(
                f"Offspring {i + 1}/{len(offspring)} "
                f"- Mean MSE: {fitness:.4f}"
            )

        # --------------------------------------------------
        # 3c. Identify elite individuals
        # --------------------------------------------------

        elite_indices = sorted(
            range(len(fitness_scores)),
            key=lambda i: fitness_scores[i]
        )[:config.elite_count]

        # Preserve all information associated with the elites.
        elite_fitness = [fitness_scores[i] for i in elite_indices]
        elite_trees = [best_trees[i] for i in elite_indices]
        elite_tree_mses = [best_tree_mses[i] for i in elite_indices]
        elite_scores = [evaluation_scores[i] for i in elite_indices]
        elite_seeds = [evaluation_seeds[i] for i in elite_indices]

        # --------------------------------------------------
        # 3d. Survivor selection
        # --------------------------------------------------

        population = outer_survivor_selection(
            population,
            fitness_scores,
            offspring,
            config.elite_count
        )

        # Maintain alignment between population and fitness.
        fitness_scores = elite_fitness + offspring_fitness
        best_trees = elite_trees + offspring_trees
        best_tree_mses = elite_tree_mses + offspring_tree_mses
        evaluation_scores = elite_scores + offspring_scores
        evaluation_seeds = elite_seeds + offspring_seeds

        # --------------------------------------------------
        # 3e. Update global best solution
        # --------------------------------------------------

        generation_best_index = min(
            range(len(fitness_scores)),
            key=lambda i: fitness_scores[i]
        )

        generation_best_fitness = fitness_scores[generation_best_index]

        if generation_best_fitness < best_fitness:

            best_fitness = generation_best_fitness

            best_individual = copy.deepcopy(
                population[generation_best_index]
            )

            best_tree = copy.deepcopy(
                best_trees[generation_best_index]
            )

            best_tree_mse = best_tree_mses[generation_best_index]
            best_scores = evaluation_scores[generation_best_index]
            best_seeds = evaluation_seeds[generation_best_index]

        history.append({
            "generation": generation,
            "best_fitness": best_fitness
        })

        print(f"Generation {generation} complete.")
        print(f"Best fitness so far: {best_fitness:.4f}")

    # --------------------------------------------------
    # 4. Final results
    # --------------------------------------------------

    history = pd.DataFrame(history)

    best_equation = print_tree(best_tree)

    print("\n========== OUTER EA COMPLETE ==========")

    print("\nBest GP Configuration:")

    for parameter, value in best_individual.items():
        print(f"  {parameter}: {value}")

    print(f"\nMean Fitness: {best_fitness:.6f}")

    print("\nBest Evolved Equation:")
    print(best_equation)

    print(f"Equation MSE: {best_tree_mse:.6f}")

    # Save experiment results.
    save_results(
        config,
        best_individual,
        best_fitness,
        best_tree,
        best_tree_mse,
        best_scores,
        best_seeds,
        history
    )

    return best_individual, best_fitness, history


# --------------------------------------------------
# Main execution
# --------------------------------------------------

if __name__ == "__main__":

    outer_config = OuterConfig()

    best_individual, best_fitness, history = run_outer_ea(
        config=outer_config,
        seed=42
    )

    print("\nFitness History:")
    print(history)
