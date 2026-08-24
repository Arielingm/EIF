"""Ventaneo de un experimento completo + estandarización z-score."""

import os
import numpy as np
import pandas as pd

from utils.config import (
    VENTANA, SENALES_ELECTRICAS, SENALES_VIBRACION,
    N_BINS_ELEC, N_BINS_VIB, NOMBRES_COLS,
)
from utils.data import cargar_senales
from utils.features import espectro_db, envolvente_espectro_db


def procesar_archivo(args):
    """
    Para cada ventana de 1s del experimento: extrae espectro_db (eléctricas)
    o envolvente_espectro_db (vibración) y recorta a [0, F_MAX] Hz.
    Devuelve (archivo, filas, n_ventanas, error).
    """
    filepath, archivo = args
    try:
        senales    = cargar_senales(filepath)
        n_muestras = len(next(iter(senales.values())))
        n_ventanas = n_muestras // VENTANA
        filas      = []

        for v in range(n_ventanas):
            inicio = v * VENTANA
            fin    = inicio + VENTANA
            fila   = []

            for nombre in SENALES_ELECTRICAS:
                segmento = senales[nombre][inicio:fin].astype(np.float64)
                fila.extend(espectro_db(segmento, N_BINS_ELEC))

            for nombre in SENALES_VIBRACION:
                segmento = senales[nombre][inicio:fin].astype(np.float64)
                fila.extend(envolvente_espectro_db(segmento, N_BINS_VIB))

            filas.append(fila)

        return archivo, filas, n_ventanas, None

    except Exception as e:
        return archivo, [], 0, str(e)


def calcular_y_guardar_estadisticos(df: pd.DataFrame, path_csv: str) -> pd.DataFrame:
    """Media y desviación típica columna a columna, calculadas SOLO sobre train (sanos)."""
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    stats = pd.DataFrame({"mean": df.mean(), "std": df.std(ddof=1)})
    stats.to_csv(path_csv)
    return stats


def cargar_estadisticos(path_csv: str) -> pd.DataFrame:
    return pd.read_csv(path_csv, index_col=0)


def estandarizar(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Z-score usando las medias/std del conjunto de entrenamiento (sanos).
    Nunca se recalculan sobre val/test para evitar fuga de información."""
    return (df - stats["mean"].values) / stats["std"].values
