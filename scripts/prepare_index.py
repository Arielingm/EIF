import pandas as pd
import numpy as np

# =============================
# CONFIGURACIÓN
# =============================

CSV_PATH    = "Motor_DB/index/master_index.csv"
RANDOM_SEED = 42

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
# CARGAR CSV
# =============================

df = pd.read_csv(CSV_PATH)

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
# ASIGNAR SPLIT
# =============================

df["Split"] = ""

# Sanos → 70/15/15
sanos_idx = df[df["Maquina"] == "h"].index.tolist()

np.random.seed(RANDOM_SEED)
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

# Fallos → todos a test
fallos_idx = df[df["Maquina"] != "h"].index
df.loc[fallos_idx, "Split"] = "test"

# =============================
# GUARDAR CSV ACTUALIZADO
# =============================

df.to_csv(CSV_PATH, index=False)
print(f"CSV actualizado: {CSV_PATH}\n")

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

# Fallos por etiqueta
print("\n--- FALLOS ---")
fallos = df[df["Maquina"] != "h"].groupby("Fallo")
for fallo, grupo in sorted(fallos):
    print(f"  {fallo:<50}: {len(grupo)} archivos")

print()
print(f"  TOTAL sanos : {len(df[df['Maquina'] == 'h'])} archivos")
print(f"  TOTAL fallos: {len(df[df['Maquina'] != 'h'])} archivos")
print(f"  TOTAL general: {len(df)} archivos")
print("=" * 55)