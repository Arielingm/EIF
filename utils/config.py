"""
Configuración compartida por todos los notebooks del TFM (01-05).

Centraliza rutas, constantes de señal y criterios de filtrado que antes
estaban duplicados en cada script/notebook (Motor_DB/index/master_index.csv
ya es la fuente de verdad de qué experimento es relevante: ver columna
Split, que vale 'excluido' para transitorios, S20 y control grid directo).
"""

import random
import numpy as np

# =============================
# RUTAS
# =============================

CSV_INDEX      = "Motor_DB/index/master_index.csv"
RUTA_DATA      = "Motor_DB/data"
RUTA_FEATURES  = "features_fft"
PATH_MEDIAS    = "features_fft/medias_sanos.csv"
RUTA_RESULTADOS = "resultados"

# =============================
# SEÑAL
# =============================

FS      = 20000       # Hz
VENTANA = FS * 1       # ventanas de 1 segundo = 20000 muestras

SENALES_ELECTRICAS = ["u", "v", "w"]
SENALES_VIBRACION  = ["front_DE_Y", "front_DE_Z",
                       "rear_NDE_Y", "rear_NDE_Z",
                       "housing"]
TODAS_SENALES = SENALES_ELECTRICAS + SENALES_VIBRACION

# F_MAX diferenciado por tipo de señal:
#   Eléctricas : 200 Hz  — armónicos de red (50, 100, 150, 200 Hz), MCSA clásico
#   Vibración  : 1000 Hz — frecuencias de rodamiento (BPFO/BPFI/BSF/FTF) y barra rota
F_MAX_ELEC = 200
F_MAX_VIB  = 1000
N_BINS_ELEC = int(F_MAX_ELEC * VENTANA / FS) + 1   # 201 bins
N_BINS_VIB  = int(F_MAX_VIB  * VENTANA / FS) + 1   # 1001 bins

NOMBRES_COLS_ELEC = [f"{s}_bin{i}" for s in SENALES_ELECTRICAS for i in range(N_BINS_ELEC)]
NOMBRES_COLS_VIB  = [f"{s}_bin{i}" for s in SENALES_VIBRACION  for i in range(N_BINS_VIB)]
NOMBRES_COLS      = NOMBRES_COLS_ELEC + NOMBRES_COLS_VIB

VENTANAS_POR_EXP = 100   # 100 segundos a 20kHz -> 100 ventanas de 1s por experimento

# =============================
# ETIQUETAS DEL ÍNDICE
# =============================

MACHINE_LABELS = {
    "h": "sano",
    "e": "fallo_barra_rota_1",
    "b": "fallo_barra_rota",
    "v": "fallo_multiples_barras",
    "p": "prognosis_barra_rota",
    "g": "fallo_rodamiento_groove",
    "o": "fallo_rodamiento_agujero_6mm",
    "r": "fallo_rodamiento_agujero_3mm",
    "c": "fallo_rodamiento_real",
}

CONTROL_LABELS = {
    "l": "grid",
    "s": "scalar",
    "d": "DTC",
}

# =============================
# CRITERIOS DE EXCLUSIÓN DEL SPLIT
# =============================
# Fuente única de verdad para scripts/prepare_index.py (que las aplica al
# regenerar Motor_DB/index/master_index.csv) y para cualquier análisis que
# necesite replicar el mismo filtro sin releer el índice ya filtrado.

# Regímenes estables: excluye transitorios (STrap=rampa, SSteps=escalonado,
# SStart=arranque) y S20 (revolución demasiado baja, no representativa).
VELOCIDADES_ESTABLES = {"S1500", "S1200", "S900"}

# Control 'l' (grid directo) excluido: representación insuficiente en train
# para que el modelo aprenda qué es normal con conexión directa a red.
CONTROL_EXCLUIDO = {"l"}

# =============================
# REPRODUCIBILIDAD
# =============================

SEED = 42


def fijar_semillas(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
