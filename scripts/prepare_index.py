import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from utils.config import (
    CSV_INDEX, SEED, MACHINE_LABELS, CONTROL_LABELS,
    VELOCIDADES_ESTABLES, CONTROL_EXCLUIDO,
)

# =============================
# CARGAR CSV
# =============================

df = pd.read_csv(CSV_INDEX)

# =============================
# ASIGNAR ETIQUETA DE FALLO
# =============================

def construir_etiqueta(row):
    maquina  = str(row["Maquina"]).strip()
    control  = str(row["Control"]).strip()
    velocidad = str(row["Velocidad"]).strip()

    if maquina == "h":
        return "sano"

    tipo    = MACHINE_LABELS.get(maquina, f"fallo_{maquina}")
    ctrl    = CONTROL_LABELS.get(control, control)

    return f"{tipo}_{ctrl}_{velocidad}"

df["Fallo"] = df.apply(construir_etiqueta, axis=1)

# =============================
# MARCAR EXPERIMENTOS IRRELEVANTES
# =============================
# Se excluyen del split (no se usan ni para entrenar ni para evaluar),
# pero se conservan en el CSV con el motivo para trazabilidad.
# Criterios definidos en utils/config.py (VELOCIDADES_ESTABLES, CONTROL_EXCLUIDO).

def motivo_exclusion(row):
    motivos = []
    if str(row["Velocidad"]).strip() not in VELOCIDADES_ESTABLES:
        motivos.append("transitorio")
    if str(row["Control"]).strip() in CONTROL_EXCLUIDO:
        motivos.append("control_grid_directo")
    return ",".join(motivos)

df["Motivo_Exclusion"] = df.apply(motivo_exclusion, axis=1)
es_relevante = df["Motivo_Exclusion"] == ""

# =============================
# ASIGNAR SPLIT
# =============================

df["Split"] = ""
df.loc[~es_relevante, "Split"] = "excluido"

# Sanos relevantes → 70/15/15
sanos_idx = df[(df["Maquina"] == "h") & es_relevante].index.tolist()

np.random.seed(SEED)
np.random.shuffle(sanos_idx)

n_total = len(sanos_idx)
n_train = int(n_total * 0.70)
n_val   = int(n_total * 0.15)

train_idx = sanos_idx[:n_train]
val_idx   = sanos_idx[n_train:n_train + n_val]
test_idx  = sanos_idx[n_train + n_val:]

df.loc[train_idx, "Split"] = "train"
df.loc[val_idx,   "Split"] = "val"
df.loc[test_idx,  "Split"] = "test"

# Fallos relevantes → todos a test
fallos_idx = df[(df["Maquina"] != "h") & es_relevante].index
df.loc[fallos_idx, "Split"] = "test"

# =============================
# GUARDAR CSV ACTUALIZADO
# =============================

df.to_csv(CSV_INDEX, index=False)
print(f"CSV actualizado: {CSV_INDEX}\n")

# =============================
# RESUMEN
# =============================

print("=" * 55)
print("RESUMEN DE LA BASE DE DATOS")
print("=" * 55)

# Sanos por split
print("\n--- SANOS ---")
for split in ["train", "val", "test"]:
    n = len(df[(df["Maquina"] == "h") & (df["Split"] == split)])
    print(f"  sano {split:<6}: {n} archivos")

# Fallos por etiqueta (solo relevantes, incluidos en test)
print("\n--- FALLOS (relevantes) ---")
fallos = df[(df["Maquina"] != "h") & es_relevante].groupby("Fallo")
for fallo, grupo in sorted(fallos):
    print(f"  {fallo:<50}: {len(grupo)} archivos")

# Excluidos y motivo
print("\n--- EXCLUIDOS DEL SPLIT ---")
excluidos = df[~es_relevante]
if len(excluidos) == 0:
    print("  (ninguno)")
else:
    print(excluidos["Motivo_Exclusion"].value_counts().rename("n_archivos").to_string())

print()
print(f"  TOTAL sanos     : {len(df[df['Maquina'] == 'h'])} archivos ({len(df[(df['Maquina'] == 'h') & es_relevante])} relevantes)")
print(f"  TOTAL fallos    : {len(df[df['Maquina'] != 'h'])} archivos ({len(df[(df['Maquina'] != 'h') & es_relevante])} relevantes)")
print(f"  TOTAL excluidos : {len(excluidos)} archivos")
print(f"  TOTAL general   : {len(df)} archivos")
print("=" * 55)
