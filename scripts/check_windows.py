import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from utils.config import CSV_INDEX, RUTA_FEATURES

index = pd.read_csv(CSV_INDEX)

print("=" * 65)
print("VENTANAS POR EXPERIMENTO")
print("=" * 65)

for split in ["train", "val", "test"]:
    carpeta = os.path.join(RUTA_FEATURES, split)
    if not os.path.exists(carpeta):
        continue

    print(f"\n--- {split.upper()} ---")

    for archivo in sorted(os.listdir(carpeta)):
        if not archivo.endswith(".csv"):
            continue

        ruta = os.path.join(carpeta, archivo)
        df   = pd.read_csv(ruta)

        # Buscar cuántos archivos originales componen este CSV
        nombre_fallo = archivo.replace(".csv", "")
        if split == "train":
            n_archivos = len(index[index["Split"] == "train"])
        else:
            n_archivos = len(index[
                (index["Split"] == split) &
                (index["Fallo"] == nombre_fallo)
            ])

        n_ventanas_total = df.shape[0]
        ventanas_por_archivo = n_ventanas_total / n_archivos if n_archivos > 0 else 0

        print(f"  {archivo:<55} "
              f"filas={n_ventanas_total:>5} | "
              f"archivos={n_archivos:>3} | "
              f"ventanas/exp={ventanas_por_archivo:>7.1f}")

print("\n" + "=" * 65)