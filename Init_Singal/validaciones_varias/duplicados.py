import os
import h5py
import hashlib
import numpy as np
import pandas as pd

data_folder = "Motor_DB/data"

# almacenar hashes
hash_dict = {}

files = [f for f in os.listdir(data_folder) if f.endswith(".mat")]

print(f"\nAnalizando {len(files)} archivos...\n")

for file in files:

    path = os.path.join(data_folder, file)

    try:

        with h5py.File(path, 'r') as f:

            u = np.array(f["test/signals/electrical/u"])
            v = np.array(f["test/signals/electrical/v"])
            w = np.array(f["test/signals/electrical/w"])

            front = np.array(f["test/signals/vibration/front_DE"])
            rear = np.array(f["test/signals/vibration/rear_NDE"])

            housing = np.array(f["test/signals/vibration/housing_uniaxial"])
            speed = np.array(f["test/signals/speed_raw"])

        # concatenar todas las señales
        combined = np.concatenate([
            u.flatten(),
            v.flatten(),
            w.flatten(),
            front.flatten(),
            rear.flatten(),
            housing.flatten(),
            speed.flatten()
        ])

        # crear hash
        hash_val = hashlib.sha256(combined.tobytes()).hexdigest()

        if hash_val not in hash_dict:
            hash_dict[hash_val] = []

        hash_dict[hash_val].append(file)

    except Exception as e:
        print(f"Error leyendo {file}: {e}")


# =============================
# BUSCAR DUPLICADOS
# =============================

duplicates = []

for h, file_list in hash_dict.items():

    if len(file_list) > 1:

        for f in file_list:
            duplicates.append({
                "hash": h,
                "archivo": f
            })


dup_df = pd.DataFrame(duplicates)

# guardar reporte
dup_df.to_csv("duplicate_signals_report.csv", index=False)


# =============================
# RESULTADOS
# =============================

print("\n========================")
print("RESULTADO")
print("========================")

num_duplicates = sum(len(v)-1 for v in hash_dict.values() if len(v) > 1)

print("Archivos totales:", len(files))
print("Experimentos duplicados:", num_duplicates)

if num_duplicates > 0:
    print("\nDuplicados encontrados. Ver:")
    print("duplicate_signals_report.csv")
else:
    print("\nNo se encontraron duplicados.")