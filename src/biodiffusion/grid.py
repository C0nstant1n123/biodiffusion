import numpy as np
from typing import List, Optional

class BioGrid:
    """Gère une grille 2D pour la simulation de diffusion moléculaire stochastique."""

    def __init__(self, 
                 width: int = 10, 
                 height: int = 10, 
                 molecule_names: Optional[List[str]] = None, 
                 spatial_scale: float = 1.0):
        """Initialise la grille biologique.

        Args:
            width (int): Nombre de cases en largeur. Défaut à 10.
            height (int): Nombre de cases en hauteur. Défaut à 10.
            molecule_names (Optional[List[str]]): Liste des noms des molécules (ex: ['AHL']).
            spatial_scale (float): Taille physique d'un carreau (dx). Défaut à 1.0.
        """
        self.width = width
        self.height = height
        self.dx = spatial_scale

        self.mol_indices = {name: i for i, name in enumerate(molecule_names)} if molecule_names else {}
        self.num_species = len(self.mol_indices)

        # Grille 3D : Hauteur x Largeur x Espèces
        self.grid = np.zeros((height, width, self.num_species), dtype=int)
        self.diffusion_coeffs = {name: 0.0 for name in self.mol_indices}

    def add_molecule(self, x: int, y: int, molecule_name: str, amount: int) -> None:
        """Ajoute (ou retire) des molécules à une position spécifique.

        Cette fonction met à jour la grille en ajoutant la quantité spécifiée.
        Si la quantité résultante est négative, elle est ramenée à 0 (clamping).

        Args:
            x (int): Coordonnée x de la case (0 à width-1).
            y (int): Coordonnée y de la case (0 à height-1).
            molecule_name (str): Nom de la molécule cible.
            amount (int): Quantité à ajouter (négatif pour retirer).

        Raises:
            KeyError: Si le nom de la molécule n'existe pas.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = self.mol_indices[molecule_name]
            current_val = self.grid[y, x, idx]
            new_val = current_val + int(amount)
            self.grid[y, x, idx] = max(0, new_val)

    def set_diffusion_coefficient(self, molecule_name: str, D: float) -> None:
        """Définit le coefficient de diffusion pour une espèce donnée.

        Args:
            molecule_name (str): Nom de la molécule.
            D (float): Coefficient de diffusion (unités de carreau^2/s).
        """
        if molecule_name in self.diffusion_coeffs:
            self.diffusion_coeffs[molecule_name] = D

    def diffuse(self, tau_diff: float) -> None:
        """Exécute une étape de simulation stochastique de la diffusion.

        Utilise une méthode binomiale pour simuler le mouvement brownien
        vers les cases adjacentes (Haut, Bas, Gauche, Droite).

        Args:
            tau_diff (float): Pas de temps de la simulation (secondes).
        """
        # Sécurité : on s'assure qu'il n'y a pas de valeurs négatives
        self.grid[self.grid < 0] = 0
        
        # Copie pour calculer les flux sans modifier la grille en cours de lecture
        current_grid = self.grid.copy()
        next_grid = self.grid.copy()

        for mol_name, idx in self.mol_indices.items():
            D = self.diffusion_coeffs[mol_name]

            if D <= 0:
                continue

            # Correction mathématique : parenthèses ajoutées au dénominateur
            # Probabilité de bouger dans UNE direction spécifique (1/4 du total)
            prob_move = (D * tau_diff) / (4 * (self.dx ** 2))

            # Limite de stabilité pour la simulation
            if prob_move > 0.25:
                prob_move = 0.25

            species_current = current_grid[:, :, idx]

            # --- FLUX DROITE (Right) ---
            # On prend tout sauf la dernière colonne, et on regarde ceux qui vont à droite
            movers = np.random.binomial(species_current[:, :-1], prob_move)
            next_grid[:, :-1, idx] -= movers 
            next_grid[:, 1:, idx]  += movers 

            # --- FLUX GAUCHE (Left) ---
            movers = np.random.binomial(species_current[:, 1:], prob_move)
            next_grid[:, 1:, idx]  -= movers
            next_grid[:, :-1, idx] += movers

            # --- FLUX BAS (Down) ---
            movers = np.random.binomial(species_current[:-1, :], prob_move)
            next_grid[:-1, :, idx] -= movers
            next_grid[1:, :, idx]  += movers

            # --- FLUX HAUT (Up) ---
            movers = np.random.binomial(species_current[1:, :], prob_move)
            next_grid[1:, :, idx]  -= movers
            next_grid[:-1, :, idx] += movers

        self.grid = next_grid

    def get_concentration(self, x: int, y: int, molecule_name: str) -> int:
        """Récupère la quantité de molécules à une position donnée.

        Args:
            x (int): Coordonnée x.
            y (int): Coordonnée y.
            molecule_name (str): Nom de la molécule.

        Returns:
            int: La quantité présente sur la case.
        """
        idx = self.mol_indices[molecule_name]
        return self.grid[y, x, idx]

