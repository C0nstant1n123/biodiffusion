import pytest
import numpy as np
from biodiffusion import BioGrid

# --- Tests pour add_molecule ---

def test_add_molecule_basic():
    """Vérifie qu'on peut ajouter des molécules à une position précise."""
    grid = BioGrid(width=5, height=5, molecule_names=["AHL"])
    grid.add_molecule(2, 3, "AHL", 100)
    
    assert grid.get_concentration(2, 3, "AHL") == 100

def test_add_molecule_negative():
    """Vérifie que la concentration ne tombe pas sous zéro."""
    grid = BioGrid(width=5, height=5, molecule_names=["AHL"])
    grid.add_molecule(0, 0, "AHL", 10)
    grid.add_molecule(0, 0, "AHL", -50) 
    
    assert grid.get_concentration(0, 0, "AHL") == 0

# --- Tests pour diffuse ---

def test_diffusion_conservation():
    """Vérifie que la diffusion conserve le nombre total de molécules (système fermé)."""
    grid = BioGrid(width=10, height=10, molecule_names=["AHL"])
    grid.set_diffusion_coefficient("AHL", 0.1)
    grid.add_molecule(5, 5, "AHL", 1000) # On met 1000 molécules au centre
    
    initial_sum = np.sum(grid.grid)
    grid.diffuse(tau_diff=0.1)
    final_sum = np.sum(grid.grid)
    
    assert initial_sum == final_sum

def test_diffusion_movement():
    """Vérifie que les molécules bougent effectivement après diffusion."""
    grid = BioGrid(width=10, height=10, molecule_names=["AHL"])
    grid.set_diffusion_coefficient("AHL", 1.0) # Forte diffusion
    grid.add_molecule(5, 5, "AHL", 500)
    
    # Avant diffusion, le centre a 500, les autres 0
    assert grid.get_concentration(5, 5, "AHL") == 500
    
    grid.diffuse(tau_diff=1.0)
    
    # Après diffusion, la case centrale devrait avoir perdu des molécules
    assert grid.get_concentration(5, 5, "AHL") < 500
    # Et la somme totale doit toujours être 500
    assert np.sum(grid.grid) == 500
