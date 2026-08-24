"""Agregación de scores por experimento y métricas de detección de anomalías,
comunes a los notebooks 02 (Isolation Forest), 03 (EIF) y 04 (Transformers)
para que el notebook 05 pueda comparar resultados de forma homogénea."""

import json
import os

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

from utils.config import RUTA_RESULTADOS


def agregar_por_experimento(scores: np.ndarray, ventanas_por_exp: int) -> np.ndarray:
    """Divide los scores en bloques de `ventanas_por_exp` y devuelve la
    mediana de cada bloque (1 score por experimento)."""
    medianas = []
    for i in range(0, len(scores), ventanas_por_exp):
        grupo = scores[i:i + ventanas_por_exp]
        if len(grupo) > 0:
            medianas.append(np.median(grupo))
    return np.array(medianas)


def agregar_variable(scores: np.ndarray, n_archivos: int) -> np.ndarray:
    """Para grupos donde el nº de ventanas no es múltiplo exacto de
    ventanas_por_exp (experimentos truncados): reparte el total entre
    n_archivos a partes iguales."""
    if n_archivos == 0:
        return np.array([])
    ventanas_por_exp = len(scores) // n_archivos
    return agregar_por_experimento(scores[:ventanas_por_exp * n_archivos], ventanas_por_exp)


def calcular_metricas(y_true: np.ndarray, y_score: np.ndarray, umbral: float) -> dict:
    """
    y_true  : 1 = fallo, 0 = sano (por experimento)
    y_score : score de anomalía por experimento (mayor = más anómalo)
    umbral  : score a partir del cual se predice fallo
    """
    y_pred = (y_score > umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")   # solo una clase presente

    return {
        "umbral":    float(umbral),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "auc":       float(auc),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_experimentos": int(len(y_true)),
    }


def guardar_resultado(resultado: dict, nombre_archivo: str, ruta_resultados: str = RUTA_RESULTADOS):
    """Guarda `resultado` en resultados/<nombre_archivo>.json (convención:
    <modelo>_<fuente_senal>.json, p.ej. eif_hibrido.json)."""
    os.makedirs(ruta_resultados, exist_ok=True)
    path = os.path.join(ruta_resultados, nombre_archivo)
    with open(path, "w") as f:
        json.dump(resultado, f, indent=2)
    return path
