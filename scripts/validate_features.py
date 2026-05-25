import os
import numpy as np
import pandas as pd

# =============================
# CONFIGURACIÓN
# =============================

RUTA_FEATURES   = "features"
CSV_INDEX       = "Motor_DB/index/master_index.csv"
PATH_MEDIAS     = "features/medias_sanos.csv"
N_FEATURES      = 48


NOMBRES_COLS = [
    f"{s}_{f}"
    for s in ["u", "v", "w", "front_DE_Y", "front_DE_Z",
              "rear_NDE_Y", "rear_NDE_Z", "housing"]
    for f in ["katz", "perm_entropy", "kurtosis",
              "rms", "pico", "cresta"]
]
# =============================
# HELPERS
# =============================

ok_count   = 0
warn_count = 0
err_count  = 0

def OK(msg):
    global ok_count
    ok_count += 1
    print(f"  ✅ OK      {msg}")

def WARN(msg):
    global warn_count
    warn_count += 1
    print(f"  ⚠️  WARN   {msg}")

def ERR(msg):
    global err_count
    err_count += 1
    print(f"  ❌ ERROR   {msg}")

def cargar_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        ERR(f"No se pudo leer {path}: {e}")
        return None

# =============================
# VALIDACIÓN 1 — ESTRUCTURA
# =============================

def validar_estructura(index):
    print("\n" + "="*60)
    print("VALIDACIÓN 1: ESTRUCTURA DE ARCHIVOS")
    print("="*60)

    # Verificar medias_sanos.csv
    if os.path.exists(PATH_MEDIAS):
        medias = pd.read_csv(PATH_MEDIAS, header=None)
        if len(medias) == N_FEATURES:
            OK(f"medias_sanos.csv existe y tiene {N_FEATURES} filas")
        else:
            ERR(f"medias_sanos.csv tiene {len(medias)} filas, esperado {N_FEATURES}")
    else:
        ERR("medias_sanos.csv NO existe")

    # Verificar carpetas
    for carpeta in ["train", "val", "test"]:
        ruta = os.path.join(RUTA_FEATURES, carpeta)
        if os.path.exists(ruta):
            csvs = [f for f in os.listdir(ruta) if f.endswith('.csv')]
            OK(f"Carpeta {carpeta}/ existe con {len(csvs)} CSV(s)")
        else:
            ERR(f"Carpeta {carpeta}/ NO existe")

    # Verificar que todos los grupos del index tienen su CSV
    print("\n  Verificando CSVs por grupo...")
    for split in ["val", "test"]:
        grupos = index[index["Split"] == split]["Fallo"].unique()
        for fallo in sorted(grupos):
            ruta_csv = os.path.join(RUTA_FEATURES, split, f"{fallo}.csv")
            if os.path.exists(ruta_csv):
                OK(f"{split}/{fallo}.csv existe")
            else:
                ERR(f"{split}/{fallo}.csv NO existe")

    # train
    ruta_train = os.path.join(RUTA_FEATURES, "train", "sano_train.csv")
    if os.path.exists(ruta_train):
        OK("train/sano_train.csv existe")
    else:
        ERR("train/sano_train.csv NO existe")

# =============================
# VALIDACIÓN 2 — COLUMNAS Y NANS
# =============================

def validar_columnas_y_nans():
    print("\n" + "="*60)
    print("VALIDACIÓN 2: COLUMNAS Y NaNs")
    print("="*60)

    for split in ["train", "val", "test"]:
        carpeta = os.path.join(RUTA_FEATURES, split)
        if not os.path.exists(carpeta):
            continue
        for archivo in sorted(os.listdir(carpeta)):
            if not archivo.endswith('.csv'):
                continue
            ruta = os.path.join(carpeta, archivo)
            df = cargar_csv(ruta)
            if df is None:
                continue

            # Columnas
            if df.shape[1] == N_FEATURES:
                OK(f"{split}/{archivo}: {df.shape[1]} columnas, {df.shape[0]} filas")
            else:
                ERR(f"{split}/{archivo}: {df.shape[1]} columnas (esperado {N_FEATURES})")

            # NaNs
            n_nan = df.isna().sum().sum()
            if n_nan == 0:
                OK(f"{split}/{archivo}: sin NaNs")
            else:
                ERR(f"{split}/{archivo}: {n_nan} NaNs detectados")

            # Infinitos
            n_inf = np.isinf(df.values).sum()
            if n_inf == 0:
                OK(f"{split}/{archivo}: sin infinitos")
            else:
                ERR(f"{split}/{archivo}: {n_inf} infinitos detectados")

            # Filas vacías
            if df.shape[0] > 0:
                OK(f"{split}/{archivo}: no está vacío")
            else:
                ERR(f"{split}/{archivo}: está VACÍO")

# =============================
# VALIDACIÓN 3 — ESTANDARIZACIÓN
# =============================

def validar_estandarizacion():
    print("\n" + "="*60)
    print("VALIDACIÓN 3: ESTANDARIZACIÓN")
    print("="*60)

    # Train debe tener media ≈ 0 y std ≈ 1
    print("\n  --- TRAIN (media debe ser ≈0, std ≈1) ---")
    ruta_train = os.path.join(RUTA_FEATURES, "train", "sano_train.csv")
    df = cargar_csv(ruta_train)
    if df is not None:
        for col in df.columns:
            mean = df[col].mean()
            std  = df[col].std()
            if abs(mean) < 0.01 and abs(std - 1.0) < 0.05:
                OK(f"{col}: mean={mean:.4f}, std={std:.4f}")
            elif abs(mean) < 0.1:
                WARN(f"{col}: mean={mean:.4f}, std={std:.4f}")
            else:
                ERR(f"{col}: mean={mean:.4f}, std={std:.4f} — estandarización incorrecta")

    # Val sano debe tener medias cercanas a 0
    print("\n  --- VAL sano (media debe ser cercana a 0) ---")
    ruta_val = os.path.join(RUTA_FEATURES, "val", "sano.csv")
    df_val = cargar_csv(ruta_val)
    if df_val is not None:
        mean_global = df_val.mean().mean()
        std_global  = df_val.std().mean()
        if abs(mean_global) < 0.5:
            OK(f"Val sano: media global={mean_global:.4f}, std media={std_global:.4f}")
        else:
            WARN(f"Val sano: media global={mean_global:.4f} — algo alejada de 0")

    # Fallos deben tener medias más alejadas de 0 que los sanos
    print("\n  --- TEST fallos (medias esperadas alejadas de 0) ---")
    carpeta_test = os.path.join(RUTA_FEATURES, "test")
    if os.path.exists(carpeta_test):
        for archivo in sorted(os.listdir(carpeta_test)):
            if not archivo.endswith('.csv') or archivo == "sano.csv":
                continue
            df_fallo = cargar_csv(os.path.join(carpeta_test, archivo))
            if df_fallo is None:
                continue
            mean_global = abs(df_fallo.mean()).mean()
            if mean_global > 0.1:
                OK(f"{archivo}: desviación media={mean_global:.4f} (señal de anomalía)")
            else:
                WARN(f"{archivo}: desviación media={mean_global:.4f} (muy cerca de sano)")

# =============================
# VALIDACIÓN 4 — CONSISTENCIA CON INDEX
# =============================

def validar_consistencia(index):
    print("\n" + "="*60)
    print("VALIDACIÓN 4: CONSISTENCIA CON master_index.csv")
    print("="*60)

    for split in ["val", "test"]:
        grupos = index[index["Split"] == split].groupby("Fallo")
        for fallo, grupo in grupos:
            ruta_csv = os.path.join(RUTA_FEATURES, split, f"{fallo}.csv")
            if not os.path.exists(ruta_csv):
                continue
            df = cargar_csv(ruta_csv)
            if df is None:
                continue
            n_archivos = len(grupo)
            n_ventanas = df.shape[0]
            ventanas_por_archivo = n_ventanas / n_archivos
            if ventanas_por_archivo > 0:
                OK(f"{split}/{fallo}: {n_archivos} archivos → {n_ventanas} ventanas "
                   f"({ventanas_por_archivo:.1f} ventanas/archivo)")
            else:
                ERR(f"{split}/{fallo}: 0 ventanas para {n_archivos} archivos")

# =============================
# RESUMEN FINAL
# =============================

def resumen_final():
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"  ✅ OK    : {ok_count}")
    print(f"  ⚠️  WARN  : {warn_count}")
    print(f"  ❌ ERROR : {err_count}")

    if err_count == 0 and warn_count == 0:
        print("\n  🎉 Todo perfecto, datos listos para entrenar.")
    elif err_count == 0:
        print("\n  ⚠️  Sin errores críticos pero revisa los warnings.")
    else:
        print("\n  ❌ Hay errores críticos que deben corregirse antes de entrenar.")

    print("="*60)

# =============================
# EJECUTAR
# =============================

index = pd.read_csv(CSV_INDEX)

validar_estructura(index)
validar_columnas_y_nans()
validar_estandarizacion()
validar_consistencia(index)
resumen_final()