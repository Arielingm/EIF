import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from multiprocessing import Pool, cpu_count

from utils.config import (
    CSV_INDEX, RUTA_DATA, RUTA_FEATURES, PATH_MEDIAS,
    SENALES_ELECTRICAS, SENALES_VIBRACION, N_BINS_ELEC, N_BINS_VIB,
    F_MAX_ELEC, F_MAX_VIB, NOMBRES_COLS,
)
from utils.extraction import (
    procesar_archivo, calcular_y_guardar_estadisticos, cargar_estadisticos, estandarizar,
)

# =============================
# PIPELINE PRINCIPAL
# =============================
# Extractores usados por experimento/ventana (ver utils/features.py):
#   - espectro_db            para señales eléctricas (u, v, w)
#   - envolvente_espectro_db para señales de vibración (front/rear/housing)
# Motor_DB/index/master_index.csv (columna Split) ya es la fuente de verdad
# de qué experimentos son relevantes: Split == 'excluido' para transitorios,
# S20 y control grid directo (ver utils/config.py y scripts/prepare_index.py).


def run():

    index = pd.read_csv(CSV_INDEX)
    relevantes = index[index["Split"] != "excluido"]
    print(f"Experimentos relevantes: {len(relevantes)} de {len(index)} "
          f"({len(index) - len(relevantes)} excluidos — ver columna Motivo_Exclusion)")

    for carpeta in ["train", "val", "test"]:
        os.makedirs(os.path.join(RUTA_FEATURES, carpeta), exist_ok=True)

    n_cores = cpu_count()

    # -------------------------
    # PASO 1 — TRAIN (sanos)
    # -------------------------
    print("\n" + "="*60)
    print("PASO 1: Procesando TRAIN (sanos) — extracción de features")
    print("="*60)

    train_rows = relevantes[relevantes["Split"] == "train"]
    args_list  = [
        (os.path.join(RUTA_DATA, row["Archivo_Nuevo"]), row["Archivo_Nuevo"])
        for _, row in train_rows.iterrows()
    ]

    print(f"  {len(args_list)} archivos — {n_cores} cores\n")

    filas_train = []
    with Pool(processes=n_cores) as pool:
        resultados = pool.map(procesar_archivo, args_list)

    for archivo, filas, n_v, error in resultados:
        if error:
            print(f"  ERROR : {archivo} → {error}")
        else:
            filas_train.extend(filas)
            print(f"  OK    : {archivo} → {n_v} ventanas")

    df_train = pd.DataFrame(filas_train, columns=NOMBRES_COLS)
    print(f"\n  Shape antes de estandarizar: {df_train.shape}")

    # Calcular y guardar estadísticos (solo con train)
    stats = calcular_y_guardar_estadisticos(df_train, PATH_MEDIAS)
    df_train = estandarizar(df_train, stats)

    ruta_train = os.path.join(RUTA_FEATURES, "train", "sano_train.csv")
    df_train.to_csv(ruta_train, index=False)
    print(f"  Guardado: {ruta_train}  |  Shape: {df_train.shape}")

    # -------------------------
    # PASO 2 — VAL y TEST
    # -------------------------
    stats = cargar_estadisticos(PATH_MEDIAS)

    for split in ["val", "test"]:
        print("\n" + "="*60)
        print(f"PASO 2: Procesando {split.upper()}")
        print("="*60)

        split_rows = relevantes[relevantes["Split"] == split]

        for fallo, grupo in split_rows.groupby("Fallo"):
            args_list = [
                (os.path.join(RUTA_DATA, row["Archivo_Nuevo"]), row["Archivo_Nuevo"])
                for _, row in grupo.iterrows()
            ]

            filas_grupo = []
            with Pool(processes=n_cores) as pool:
                resultados = pool.map(procesar_archivo, args_list)

            for archivo, filas, n_v, error in resultados:
                if error:
                    print(f"  ERROR : {archivo} → {error}")
                else:
                    filas_grupo.extend(filas)
                    print(f"  OK    : {archivo} → {n_v} ventanas  [{fallo}]")

            if not filas_grupo:
                print(f"  AVISO : No hay datos para fallo '{fallo}' en {split}")
                continue

            df_grupo = pd.DataFrame(filas_grupo, columns=NOMBRES_COLS)
            df_grupo = estandarizar(df_grupo, stats)

            ruta_csv = os.path.join(RUTA_FEATURES, split, f"{fallo}.csv")
            df_grupo.to_csv(ruta_csv, index=False)
            print(f"  → Guardado: {ruta_csv}  ({df_grupo.shape[0]} ventanas)")

    # -------------------------
    # RESUMEN
    # -------------------------
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"  Features por ventana : {len(NOMBRES_COLS)}  "
      f"(eléctricas: {len(SENALES_ELECTRICAS)}×{N_BINS_ELEC} bins hasta {F_MAX_ELEC}Hz  |  "
      f"vibración: {len(SENALES_VIBRACION)}×{N_BINS_VIB} bins hasta {F_MAX_VIB}Hz)")

    for split in ["train", "val", "test"]:
        carpeta = os.path.join(RUTA_FEATURES, split)
        csvs    = sorted(f for f in os.listdir(carpeta) if f.endswith(".csv"))
        print(f"\n  {split.upper()}/")
        for a in csvs:
            df_tmp = pd.read_csv(os.path.join(carpeta, a))
            print(f"    {a:<55} {df_tmp.shape[0]:>6} ventanas")

    print("\nProcesado completado.")


# =============================
# EJECUTAR
# =============================

if __name__ == "__main__":
    run()
