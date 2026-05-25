import os
import csv
import h5py
import numpy as np
import pandas as pd
from itertools import permutations
from math import factorial, log2
from scipy.stats import entropy, kurtosis
from multiprocessing import Pool, cpu_count
import antropy as ant

# =============================
# CONFIGURACIÓN
# =============================

CSV_INDEX       = "Motor_DB/index/master_index.csv"
RUTA_ENTRADA    = "Motor_DB/data"
RUTA_SALIDA     = "features"
PATH_MEDIAS     = "features/medias_sanos.csv"

FS      = 20000
VENTANA = FS * 1  # 1 segundo = 20000 muestras

# =============================
# FEATURES
# =============================

def katz_fractal(signal):
    L = np.sum(np.sqrt(1 + np.square(np.diff(signal))))
    N = len(signal) - 1
    puntox = np.arange(1, len(signal) + 1)
    distances = np.sqrt((puntox - puntox[0])**2 + (signal - signal[0])**2)
    d = np.max(distances)
    return np.log10(N) / (np.log10(d / L) + np.log10(N))

def permutation_entropy(signal, m=6, normalizar=False):
    return ant.perm_entropy(signal, order=m, normalize=normalizar)

def extraer_features(signal):

    # Features originales
    katz  = katz_fractal(signal)
    pe    = permutation_entropy(signal)
    kurt  = kurtosis(signal) if np.std(signal) > 1e-10 else 0.0

    # Features nuevas
    rms           = np.sqrt(np.mean(signal**2))
    valor_pico    = np.max(np.abs(signal))
    factor_cresta = valor_pico / rms if rms > 1e-10 else 0.0

    return [katz, pe, kurt, rms, valor_pico, factor_cresta]
# =============================
# ESTANDARIZACIÓN
# =============================

def guardar_medias_desviaciones(medias_desv, path_csv):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, 'w', newline='') as f:
        csv.writer(f).writerows(medias_desv)

def cargar_medias_desviaciones(path_csv):
    with open(path_csv, 'r') as f:
        return [(float(fila[0]), float(fila[1])) for fila in csv.reader(f)]

def media_desviacion(values):
    mean = np.mean(values)
    std  = np.std(values, ddof=1)
    return mean, std

def zscores(values, mean, std):
    return (values - mean) / std

# =============================
# CARGA DE SEÑALES HDF5
# =============================

def cargar_señales(filepath):
    with h5py.File(filepath, 'r') as f:

        def load(path):
            arr = np.array(f[path]).squeeze()
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T
            return arr

        front = load("test/signals/vibration/front_DE")
        rear  = load("test/signals/vibration/rear_NDE")

        señales = {
            "u":          load("test/signals/electrical/u"),
            "v":          load("test/signals/electrical/v"),
            "w":          load("test/signals/electrical/w"),
            "front_DE_Y": front[:, 0] if front.ndim == 2 else front,
            "front_DE_Z": front[:, 1] if front.ndim == 2 else front,
            "rear_NDE_Y": rear[:, 0]  if rear.ndim == 2  else rear,
            "rear_NDE_Z": rear[:, 1]  if rear.ndim == 2  else rear,
            "housing":    load("test/signals/vibration/housing_uniaxial"),
        }
    return señales

# =============================
# EXTRAER FEATURES DE UN ARCHIVO
# =============================

def procesar_archivo(filepath):
    señales  = cargar_señales(filepath)
    n_muestras = len(next(iter(señales.values())))
    n_ventanas = n_muestras // VENTANA
    filas = []

    for v in range(n_ventanas):
        inicio = v * VENTANA
        fin    = inicio + VENTANA
        fila   = []
        for señal in señales.values():
            fila.extend(extraer_features(señal[inicio:fin]))
        filas.append(fila)

    return filas, n_ventanas, señales.keys()

# =============================
# NOMBRES DE COLUMNAS
# =============================

NOMBRES_COLS = [
    f"{s}_{f}"
    for s in ["u", "v", "w", "front_DE_Y", "front_DE_Z",
              "rear_NDE_Y", "rear_NDE_Z", "housing"]
    for f in ["katz", "perm_entropy", "kurtosis",
              "rms", "pico", "cresta"]
]


# =============================
# PROCESAR UN ARCHIVO (función para paralelizar)
# =============================

def procesar_archivo_wrapper(args):
    filepath, archivo = args
    try:
        señales  = cargar_señales(filepath)
        n_muestras = len(next(iter(señales.values())))
        n_ventanas = n_muestras // VENTANA
        filas = []

        for v in range(n_ventanas):
            inicio = v * VENTANA
            fin    = inicio + VENTANA
            fila   = []
            for señal in señales.values():
                fila.extend(extraer_features(señal[inicio:fin]))
            filas.append(fila)

        return archivo, filas, n_ventanas, None

    except Exception as e:
        return archivo, [], 0, str(e)

# =============================
# PIPELINE PRINCIPAL
# =============================

def run():

    index = pd.read_csv(CSV_INDEX)

    for carpeta in ["train", "val", "test"]:
        os.makedirs(os.path.join(RUTA_SALIDA, carpeta), exist_ok=True)

    # -------------------------
    # PASO 1 — TRAIN
    # -------------------------
    print("\n" + "="*55)
    print("PASO 1: Procesando TRAIN (sanos)")
    print("="*55)

    train_rows = index[index["Split"] == "train"]
    args_list  = [
        (os.path.join(RUTA_ENTRADA, row["Archivo_Nuevo"]), row["Archivo_Nuevo"])
        for _, row in train_rows.iterrows()
    ]

    filas_train = []
    n_cores = cpu_count()
    print(f"  Usando {n_cores} cores en paralelo...\n")

    with Pool(processes=n_cores) as pool:
        resultados = pool.map(procesar_archivo_wrapper, args_list)

    for archivo, filas, n_v, error in resultados:
        if error:
            print(f"  ERROR: {archivo} → {error}")
        else:
            filas_train.extend(filas)
            print(f"  OK: {archivo} → {n_v} ventanas")

    df_train = pd.DataFrame(filas_train, columns=NOMBRES_COLS)

    # Calcular y guardar medias/std
    mean_ds = []
    for col in df_train.columns:
        mean, std = media_desviacion(df_train[col].values)
        mean_ds.append((mean, std))
        df_train[col] = zscores(df_train[col].values, mean, std)

    guardar_medias_desviaciones(mean_ds, PATH_MEDIAS)

    ruta_train_csv = os.path.join(RUTA_SALIDA, "train", "sano_train.csv")
    df_train.to_csv(ruta_train_csv, index=False)
    print(f"\n  Guardado: {ruta_train_csv}")
    print(f"  Shape: {df_train.shape}")

    # -------------------------
    # PASO 2 — VAL y TEST
    # -------------------------

    medias = cargar_medias_desviaciones(PATH_MEDIAS)

    for split in ["val", "test"]:
        print("\n" + "="*55)
        print(f"PASO 2: Procesando {split.upper()}")
        print("="*55)

        split_rows = index[index["Split"] == split]

        for fallo, grupo in split_rows.groupby("Fallo"):
            args_list = [
                (os.path.join(RUTA_ENTRADA, row["Archivo_Nuevo"]), row["Archivo_Nuevo"])
                for _, row in grupo.iterrows()
            ]

            filas_grupo = []
            with Pool(processes=n_cores) as pool:
                resultados = pool.map(procesar_archivo_wrapper, args_list)

            for archivo, filas, n_v, error in resultados:
                if error:
                    print(f"  ERROR: {archivo} → {error}")
                else:
                    filas_grupo.extend(filas)
                    print(f"  OK: {archivo} → {n_v} ventanas [{fallo}]")

            if not filas_grupo:
                continue

            df_grupo = pd.DataFrame(filas_grupo, columns=NOMBRES_COLS)

            for i, col in enumerate(df_grupo.columns):
                mean, std = medias[i]
                df_grupo[col] = zscores(df_grupo[col].values, mean, std)

            ruta_csv = os.path.join(RUTA_SALIDA, split, f"{fallo}.csv")
            df_grupo.to_csv(ruta_csv, index=False)
            print(f"  → Guardado: {ruta_csv} ({df_grupo.shape[0]} ventanas)")

    # -------------------------
    # RESUMEN FINAL
    # -------------------------
    print("\n" + "="*55)
    print("RESUMEN FINAL")
    print("="*55)

    for split in ["train", "val", "test"]:
        carpeta = os.path.join(RUTA_SALIDA, split)
        archivos_csv = [f for f in os.listdir(carpeta) if f.endswith('.csv')]
        print(f"\n  {split.upper()}/")
        for a in sorted(archivos_csv):
            df_tmp = pd.read_csv(os.path.join(carpeta, a))
            print(f"    {a:<55} {df_tmp.shape[0]:>6} ventanas")

    print("\nProcesado completado.")

# =============================
# EJECUTAR
# =============================

run()