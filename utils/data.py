"""Carga de señales .mat y del índice maestro de experimentos."""

import h5py
import numpy as np
import pandas as pd

from utils.config import CSV_INDEX


def cargar_indice(csv_index: str = CSV_INDEX) -> pd.DataFrame:
    """Carga master_index.csv. La columna Split ya excluye lo irrelevante
    (transitorios, S20, control grid directo) con Split='excluido'."""
    return pd.read_csv(csv_index)


def cargar_senales(filepath: str) -> dict:
    """Carga todas las señales de un archivo .mat (formato HDF5 v7.3)."""
    with h5py.File(filepath, 'r') as f:

        def load(path):
            arr = np.array(f[path]).squeeze()
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T
            return arr

        front = load("test/signals/vibration/front_DE")
        rear  = load("test/signals/vibration/rear_NDE")

        senales = {
            "u":          load("test/signals/electrical/u"),
            "v":          load("test/signals/electrical/v"),
            "w":          load("test/signals/electrical/w"),
            "front_DE_Y": front[:, 0] if front.ndim == 2 else front,
            "front_DE_Z": front[:, 1] if front.ndim == 2 else front,
            "rear_NDE_Y": rear[:, 0]  if rear.ndim == 2  else rear,
            "rear_NDE_Z": rear[:, 1]  if rear.ndim == 2  else rear,
            "housing":    load("test/signals/vibration/housing_uniaxial"),
        }
    return senales


def cargar_velocidad(filepath: str) -> np.ndarray:
    """Carga la velocidad mecánica medida (rpm) de un archivo .mat."""
    with h5py.File(filepath, 'r') as f:
        return np.array(f["test/signals/speed_raw"]).squeeze()
