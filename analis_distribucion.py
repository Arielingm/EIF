"""
Análisis de distribución de experimentos por split, fallo y condición.
Muestra desglose completo para decidir si alguna condición está sub-representada.
"""

import pandas as pd

from utils.config import CSV_INDEX, VELOCIDADES_ESTABLES

# ============================================================
# CARGAR Y FILTRAR
# ============================================================

index = pd.read_csv(CSV_INDEX)
total_original = len(index)

index = index[index["Velocidad"].astype(str).isin(VELOCIDADES_ESTABLES)].reset_index(drop=True)

print("=" * 65)
print("DISTRIBUCIÓN DE EXPERIMENTOS")
print("=" * 65)
print(f"  Total en index       : {total_original}")
print(f"  Tras filtro estables : {len(index)}")
print(f"  Excluidos            : {total_original - len(index)}")

# ============================================================
# RESUMEN POR SPLIT
# ============================================================

print("\n" + "=" * 65)
print("RESUMEN POR SPLIT")
print("=" * 65)
print(index.groupby("Split").size().rename("n_experimentos").to_string())

# ============================================================
# TRAIN — DESGLOSE COMPLETO
# ============================================================

train = index[index["Split"] == "train"]

print("\n" + "=" * 65)
print(f"TRAIN — {len(train)} experimentos sanos")
print("=" * 65)

print(f"\n  Por Control:")
print(train["Control"].value_counts().rename("n").to_string())

print(f"\n  Por Velocidad:")
print(train["Velocidad"].value_counts().rename("n").to_string())

print(f"\n  Por Carga:")
print(train["Carga"].value_counts().rename("n").to_string())

print(f"\n  Por Generador:")
print(train["Generador"].value_counts().rename("n").to_string())

print(f"\n  Cruce Control × Velocidad:")
ct = pd.crosstab(train["Control"], train["Velocidad"])
print(ct.to_string())

print(f"\n  Cruce Control × Carga:")
ct2 = pd.crosstab(train["Control"], train["Carga"])
print(ct2.to_string())

print(f"\n  Lista completa train:")
cols = ["Archivo_Nuevo", "Generador", "Control", "Velocidad", "Carga"]
print(train[cols].sort_values(["Control", "Velocidad", "Carga"]).to_string(index=False))

# ============================================================
# VAL — DESGLOSE
# ============================================================

val = index[index["Split"] == "val"]

print("\n" + "=" * 65)
print(f"VAL — {len(val)} experimentos")
print("=" * 65)
cols_val = ["Archivo_Nuevo", "Fallo", "Control", "Velocidad", "Carga"]
print(val[cols_val].sort_values(["Fallo", "Control"]).to_string(index=False))

# ============================================================
# TEST — DESGLOSE POR FALLO
# ============================================================

test = index[index["Split"] == "test"]

print("\n" + "=" * 65)
print(f"TEST — {len(test)} experimentos")
print("=" * 65)

print(f"\n  Experimentos por grupo de fallo:")
resumen_test = (
    test.groupby("Fallo")
    .agg(
        n_exp      = ("Archivo_Nuevo", "count"),
        controles  = ("Control",   lambda x: ", ".join(sorted(x.unique()))),
        velocidades= ("Velocidad", lambda x: ", ".join(sorted(x.unique()))),
        cargas     = ("Carga",     lambda x: ", ".join(sorted(x.unique()))),
    )
    .reset_index()
)
print(resumen_test.to_string(index=False))

# ============================================================
# ALERTA — CONDICIONES SUB-REPRESENTADAS EN TRAIN
# ============================================================

print("\n" + "=" * 65)
print("ALERTA — CONDICIONES CON POCOS EXPERIMENTOS EN TRAIN")
print("=" * 65)

controles_train = train["Control"].value_counts()
for ctrl, n in controles_train.items():
    nombre = {"d": "DTC", "s": "Scalar", "l": "Grid directo"}.get(ctrl, ctrl)
    aviso  = " ← POCAS MUESTRAS" if n < 5 else ""
    print(f"  Control {ctrl} ({nombre}): {n} experimentos{aviso}")

print()
velocidades_train = train["Velocidad"].value_counts()
for vel, n in velocidades_train.items():
    aviso = " ← POCAS MUESTRAS" if n < 3 else ""
    print(f"  Velocidad {vel}: {n} experimentos{aviso}")