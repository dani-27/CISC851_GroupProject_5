# ---------------------------------------------------------------------------------------------------
# By: Ruth Carvalho, Samuel Gallo-Will, Danisa Ivanovic-Zuvic, Daniel Jeyaraj, Nabiha Naweed
# Last Updated: November 21, 2026
# Purpose:
# - Visualize GP convergence by plotting the best-so-far MSE across generations.
# - Compare experimental reaction rates with predictions from the best GP expression.
# ---------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------
# Package imports.
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Locate the project root
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create the output folder
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set a consistent appearance for presentation figures.
plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

NAVY = "#123B50"
TEAL = "#479CA6"

# ---------------------------------------------------------------------------------------------------
# 1. Plot the best-so-far MSE across generations

history = pd.read_csv(RESULTS_DIR / "gp_history.csv")

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(
    history["generation"],
    history["best_fitness"],
    color=NAVY,
    linewidth=2.5,
)

# Use a logarithmic scale to display the initial reduction in error.
ax.set_yscale("log")

ax.set_xlabel("Generation")
ax.set_ylabel("Best-so-far MSE (log scale)")
ax.set_title("Genetic programming convergence", pad=15)

ax.grid(axis="y", which="major", alpha=0.2)

final_mse = history["best_fitness"].iloc[-1]

ax.text(
    0.97,
    0.93,
    f"Final MSE = {final_mse:.2f}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    color=NAVY,
)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "gp_convergence.png",
    dpi=300,
    bbox_inches="tight",
)


# ------------------------------------------------------------
# 2. Plot the experimental data and the best GP expression


data = pd.read_csv(DATA_DIR / "Puromycin.csv")

substrate = data["conc"].to_numpy(dtype=float)
observed_rates = data["rate"].to_numpy(dtype=float)


def best_expression(S):
    """Evaluate the best expression obtained from the reported GP run.
    """
    numerator = 165.89755 * S
    denominator = 0.14386 + S

    first_term = np.divide(
        numerator,
        denominator,
        out=np.ones_like(S, dtype=float),
        where=np.abs(denominator) > 1e-10,
    )

    reciprocal_term = np.divide(
        0.39102,
        S,
        out=np.ones_like(S, dtype=float),
        where=np.abs(S) > 1e-10,
    )

    return first_term - ((S - 60.67824) + reciprocal_term)


# Calculate the fitting MSE using the original observations.
predicted_rates = best_expression(substrate)

calculated_mse = np.mean(
    (observed_rates - predicted_rates) ** 2
)

# Check that the expression and history are consistent.
if not np.isclose(calculated_mse, final_mse, rtol=1e-6):
    print(
        "Warning: the expression and history may belong "
        "to different runs."
    )

# Generate a smooth curve within the observed concentration range.
concentration_grid = np.linspace(
    substrate.min(),
    substrate.max(),
    500,
)

curve_predictions = best_expression(concentration_grid)

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.scatter(
    substrate,
    observed_rates,
    color=TEAL,
    s=65,
    edgecolors="white",
    linewidths=0.8,
    label="Experimental data",
    zorder=3,
)

ax.plot(
    concentration_grid,
    curve_predictions,
    color=NAVY,
    linewidth=2.5,
    label="Best GP expression",
)

ax.set_xlabel("Substrate concentration")
ax.set_ylabel("Reaction rate")
ax.set_title("Experimental data and learned equation", pad=15)

ax.text(
    0.97,
    0.06,
    f"Fitting MSE = {calculated_mse:.2f}",
    transform=ax.transAxes,
    ha="right",
    color=NAVY,
)

ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "gp_fitted_curve.png",
    dpi=300,
    bbox_inches="tight",
)

print(f"Figures saved in: {FIGURES_DIR}")

# Display both figures.
plt.show()