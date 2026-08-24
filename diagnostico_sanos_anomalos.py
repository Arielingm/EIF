"""
Identifica qué experimentos sanos del train/val tienen scores anómalos.
Requiere:
  - features_fft/train/sano_train.csv
  - features_fft/val/sano.csv
  - Motor_DB/index/master_index.csv
  - resultados_fft/umbral.json
"""

import numpy as np
import pandas as pd
import json
import os

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_FEATURES    = "features_fft"
CSV_INDEX        = "Motor_DB/index/master_index.csv"
RUTA_RESULTADOS  = "resultados_fft"
VENTANAS_POR_EXP = 100

# ============================================================
# CARGAR UMBRAL
# ============================================================

with open(os.path.join(RUTA_RESULTADOS, "umbral.json")) as f:
    datos_umbral = json.load(f)

umbral      = datos_umbral["umbral"]
media_sanos = datos_umbral["media"]
std_sanos   = datos_umbral["std"]

print(f"Umbral   : {umbral:.6f}")
print(f"Media    : {media_sanos:.6f}")
print(f"Std      : {std_sanos:.6f}")

# ============================================================
# CARGAR INDEX Y FILTRAR SANOS TRAIN + VAL
# ============================================================

index = pd.read_csv(CSV_INDEX)

# AÑADIR ESTO:
VELOCIDADES_ESTABLES = {"S1500", "S1200", "S900"}
index = index[index["Velocidad"].astype(str).isin(VELOCIDADES_ESTABLES)].reset_index(drop=True)

sanos_train = index[index["Split"] == "train"].reset_index(drop=True)
sanos_val   = index[index["Split"] == "val"  ].reset_index(drop=True)


print(f"\nExperimentos sanos train : {len(sanos_train)}")
print(f"Experimentos sanos val   : {len(sanos_val)}")

# ============================================================
# CARGAR SCORES  (necesitas haberlos guardado o recalcularlos)
# ============================================================
# OPCIÓN A — Si tienes los scores ya calculados en numpy/csv:
#   scores_train = np.load("resultados_fft/scores_train.npy")
#   scores_val   = np.load("resultados_fft/scores_val.npy")
#
# OPCIÓN B — Recalcular con H2O (ejecuta esto si no los tienes guardados):

import h2o
from h2o.estimators import H2OExtendedIsolationForestEstimator
import optuna

h2o.init(nthreads=-1, max_mem_size="10G")

train_h2o = h2o.import_file(os.path.join(RUTA_FEATURES, "train", "sano_train.csv"))
val_h2o   = h2o.import_file(os.path.join(RUTA_FEATURES, "val",   "sano.csv"))

# Recargar mejor modelo desde Optuna
study = optuna.load_study(
    study_name = "EIF_motores_fft_8008features",
    storage    = f"sqlite:///{RUTA_RESULTADOS}/optuna_eif_fft.db"
)
best = study.best_trial
SEED = 42

modelo = H2OExtendedIsolationForestEstimator(
    ntrees          = best.params["ntrees"],
    sample_size     = best.params["sample_size"],
    extension_level = best.params["extension_level"],
    seed            = SEED
)
modelo.train(training_frame=train_h2o)

scores_train = modelo.predict(train_h2o)["anomaly_score"].as_data_frame().values.flatten()
scores_val   = modelo.predict(val_h2o)["anomaly_score"].as_data_frame().values.flatten()

# Guardar para no recalcular
np.save(os.path.join(RUTA_RESULTADOS, "scores_train.npy"), scores_train)
np.save(os.path.join(RUTA_RESULTADOS, "scores_val.npy"),   scores_val)

print(f"\nScores train : {len(scores_train)} ventanas")
print(f"Scores val   : {len(scores_val)} ventanas")

# ============================================================
# ASIGNAR SCORES A EXPERIMENTOS Y CALCULAR MEDIANA
# ============================================================

def analizar_split(scores, index_split, split_name):
    """
    Asigna cada bloque de VENTANAS_POR_EXP scores a su experimento
    y calcula la mediana. Devuelve DataFrame ordenado por score descendente.
    """
    n_exp = len(scores) // VENTANAS_POR_EXP
    filas = []

    for i, (_, row) in enumerate(index_split.iterrows()):
        if i >= n_exp:
            break
        inicio  = i * VENTANAS_POR_EXP
        fin     = inicio + VENTANAS_POR_EXP
        bloque  = scores[inicio:fin]

        mediana  = np.median(bloque)
        max_sc   = np.max(bloque)
        n_sobre  = np.sum(bloque > umbral)   # ventanas que superan el umbral

        filas.append({
            "split":         split_name,
            "archivo":       row["Archivo_Nuevo"],
            "generador":     row.get("Generador", "?"),
            "maquina":       row.get("Maquina",   "?"),
            "control":       row.get("Control",   "?"),
            "velocidad":     row.get("Velocidad", "?"),
            "carga":         row.get("Carga",     "?"),
            "fallo":         row.get("Fallo",     "?"),
            "mediana_score": round(mediana, 6),
            "max_score":     round(max_sc,  6),
            "ventanas_sobre_umbral": int(n_sobre),
            "es_anomalo":    mediana > umbral,
        })

    return pd.DataFrame(filas).sort_values("mediana_score", ascending=False)

df_train = analizar_split(scores_train, sanos_train, "train")
df_val   = analizar_split(scores_val,   sanos_val,   "val")
df_total = pd.concat([df_train, df_val]).sort_values("mediana_score", ascending=False)

# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print("\n" + "="*65)
print("EXPERIMENTOS SANOS ANÓMALOS (mediana > umbral)")
print("="*65)

anomalos = df_total[df_total["es_anomalo"]]
print(f"\nTotal anómalos: {len(anomalos)} de {len(df_total)} experimentos sanos\n")

cols_mostrar = ["split", "archivo", "velocidad", "carga", "control",
                "mediana_score", "max_score", "ventanas_sobre_umbral"]
print(anomalos[cols_mostrar].to_string(index=False))

print("\n" + "="*65)
print("TOP 10 SANOS CON MAYOR SCORE (incluyendo no anómalos)")
print("="*65)
print(df_total[cols_mostrar].head(10).to_string(index=False))

# ============================================================
# GUARDAR
# ============================================================

df_total.to_csv(os.path.join(RUTA_RESULTADOS, "diagnostico_sanos.csv"), index=False)
print(f"\nGuardado: {RUTA_RESULTADOS}/diagnostico_sanos.csv")

# ============================================================
# RESUMEN POR CONDICIÓN
# ============================================================

print("\n" + "="*65)
print("RESUMEN — ¿En qué condición aparecen los anómalos?")
print("="*65)

for col in ["velocidad", "carga", "control"]:
    if col in anomalos.columns:
        print(f"\n  Por {col}:")
        print(anomalos[col].value_counts().to_string())