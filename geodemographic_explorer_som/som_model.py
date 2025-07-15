from minisom import MiniSom
import numpy as np
from typing import Tuple
import pandas as pd

def train_som(
    data_df: pd.DataFrame,
    x_dim: int = 25,
    y_dim: int = 25,
    sigma: float = 5,
    learning_rate: float = 0.3,
    iterations: int = 50000,
    random_seed: int = 0
) -> MiniSom:
    """
    Initialize and train a Self-Organizing Map on the given DataFrame.

    Returns the trained SOM instance.

    """
    values = data_df.drop(columns=["hex_x", "hex_y"], errors="ignore").values

    som = MiniSom(
        x_dim, y_dim,
        input_len=values.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        topology='hexagonal',
        random_seed=random_seed
    )
    som.random_weights_init(values)
    som.train_random(values, iterations)
    return som


def compute_umatrix(som: MiniSom) -> np.ndarray:
    """
    Returns the flattened U-Matrix from the trained SOM.

    """
    um = som.distance_map()
    return um.flatten()
