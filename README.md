# BioDiffusion

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 1. Project Purpose

BioDiffusion is a Python library designed to simulate **stochastic molecular diffusion** on a 2D discrete grid. 

Unlike continuous deterministic models (based on Fick's laws and PDEs), this project focuses on the microscopic behavior of particles. It models diffusion as a random walk process, making it particularly useful for **Systems Biology** education, where understanding noise and stochasticity at low molecule copy numbers is crucial.

The simulation uses a **binomial distribution** to determine particle movement, ensuring mass conservation and realistic variance over time.

## 2. Description of the Functions

The core logic resides in the `BioGrid` class. The two primary functions evaluated in this project are:

### Function A: `add_molecule(x, y, molecule_name, amount)`
This function manages the injection or removal of molecules at a specific coordinate $(x, y)$.
-  It updates the grid index corresponding to the molecule.
-  It implements a "clamping" mechanism: if a user tries to remove more molecules than present (resulting in a negative count), the value is automatically reset to 0. This guarantees physical consistency (no negative concentrations).

### Function B: `diffuse(tau_diff)`
This function computes the state of the grid at $t + \tau_{diff}$.
- For every spatial step, the function calculates a probability of movement based on the diffusion coefficient $D$ and the time step $\tau$:
  $$P_{move} = \frac{D \cdot \tau}{4 \cdot dx^2}$$ 
-  Instead of moving a fixed fraction of molecules, it uses `numpy.random.binomial`. This determines how many molecules randomly jump to adjacent cells (Up, Down, Left, Right).
-  It uses a buffer system (`current_grid` vs `next_grid`) to ensure that all fluxes are calculated simultaneously, avoiding directional bias during the update.

## 3. Installation

This project is packaged using modern Python standards (`pyproject.toml`).

### Prerequisites
- Python 3.9 or higher.
- `uv` (recommended) or standard `pip`.

### Installing via uv (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/C0nstant1n123/biodiffusion
cd biodiffusion

# 2. Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  
uv pip install -e .
```


## 4. Examples

Here is a complete example demonstrating how to set up the grid, inject molecules, and simulate diffusion. You can copy-paste this code into a `main.py` file to try it out.

```python

from biodiffusion import BioGrid
import matplotlib.pyplot as plt
import numpy as np

# 1. Initialize a 20x20 grid
grid = BioGrid(width=20, height=20, molecule_names=["AHL"], spatial_scale=1.0)

# 2. Set diffusion parameters (Diffusion coefficient D=0.1)
grid.set_diffusion_coefficient("AHL", 0.1)

# 3. Add a drop of 1000 molecules in the center (x=10, y=10)
grid.add_molecule(10, 10, "AHL", 1000)

print(f"Initial concentration at center: {grid.get_concentration(10, 10, 'AHL')}")

# Visualize the initial concentration distribution

plt.imshow(grid.grid, cmap='viridis', origin='lower')
plt.colorbar(label='Concentration of AHL')
plt.title('Concentration Distribution of AHL after Diffusion')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Run the simulation multiple times to observe diffusion behavior

for _ in range(5):
    # Run the simulation for 100 time steps
    for _ in range(100):
        grid.diffuse(tau_diff=0.5)

    # 5. Check results
    center_conc = grid.get_concentration(10, 10, "AHL")
    total_mass = np.sum(grid.grid)

    print(f"Concentration at center after diffusion: {center_conc}")
    print(f"Total mass (should be approx 1000): {total_mass}")


    # 6. Visualize the concentration distribution

    plt.imshow(grid.grid, cmap='viridis', origin='lower')
    plt.colorbar(label='Concentration of AHL')
    plt.title('Concentration Distribution of AHL after Diffusion')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

```

## 5. Run Pytest

This project uses `pytest` to ensure reliability, mass conservation, and correct grid boundaries.

### Using uv (Recommended)

If you have installed the project with `uv`, simply run:

```bash
uv run pytest
```

### Using standard Python

If you are in a standard virtual environment:

```bash
pytest
```