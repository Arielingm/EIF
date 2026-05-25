import pandas as pd
import numpy as np
import h5py
from scipy.io import loadmat
import os

# =============================
# RUTAS
# =============================

csv_path = "Motor_DB/index/master_index.csv"
original_folder = "DataBase_original"
processed_folder = "Motor_DB/data"

# =============================
# FUNCION DE COMPARACION ROBUSTA
# =============================

def equal_signal(a, b):

    a = np.squeeze(a)
    b = np.squeeze(b)

    if a.shape != b.shape:

        if a.T.shape == b.shape:
            a = a.T

        elif b.T.shape == a.shape:
            b = b.T

        else:
            return False

    return np.allclose(a, b, atol=1e-10)


# =============================
# CARGAR CSV
# =============================

df = pd.read_csv(csv_path)

total = len(df)
ok = 0
errors = 0

report = []

print(f"\nVerificando {total} archivos...\n")

# =============================
# LOOP PRINCIPAL
# =============================

for _, row in df.iterrows():

    original_file = os.path.join(original_folder, row["ID_Original"])
    new_file = os.path.join(processed_folder, row["Archivo_Nuevo"])

    try:

        # =========================
        # CARGAR ORIGINAL
        # =========================

        original = loadmat(original_file)

        iu = original["iu"]
        iv = original["iv"]
        iw = original["iw"]

        AccDEY = original["AccDEY"]
        AccDEZ = original["AccDEZ"]

        AccNDEY = original["AccNDEY"]
        AccNDEZ = original["AccNDEZ"]

        a = original["a"]
        s = original["s"]

        # =========================
        # CARGAR PROCESADO (HDF5)
        # =========================

        with h5py.File(new_file, 'r') as f:

            u_new = np.array(f["test/signals/electrical/u"])
            v_new = np.array(f["test/signals/electrical/v"])
            w_new = np.array(f["test/signals/electrical/w"])

            front = np.array(f["test/signals/vibration/front_DE"])
            rear = np.array(f["test/signals/vibration/rear_NDE"])

            # corregir orientación MATLAB/HDF5
            if front.shape[0] == 2:
                front = front.T

            if rear.shape[0] == 2:
                rear = rear.T

            housing = np.array(f["test/signals/vibration/housing_uniaxial"])
            speed = np.array(f["test/signals/speed_raw"])

        # Deploy Validation
        # print("iu shape:", np.squeeze(iu).shape, "u_new shape:", np.squeeze(u_new).shape)
        # print("a shape:", np.squeeze(a).shape, "housing shape:", np.squeeze(housing).shape)
        # print("s shape:", np.squeeze(s).shape, "speed shape:", np.squeeze(speed).shape)
        # print("front shape:", front.shape)
        # print("rear shape:", rear.shape)

        # for name, orig, new in [
        #     ("iu", np.squeeze(iu), np.squeeze(u_new)),
        #     ("iv", np.squeeze(iv), np.squeeze(v_new)),
        #     ("iw", np.squeeze(iw), np.squeeze(w_new)),
        #     ("a",  np.squeeze(a),  np.squeeze(housing)),
        #     ("s",  np.squeeze(s),  np.squeeze(speed)),
        # ]:
        #     diff = np.abs(orig - new)
        #     print(f"{name}: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}")
        
        # =========================
        # COMPARACIONES
        # =========================

        checks = {

            "iu": equal_signal(iu, u_new),
            "iv": equal_signal(iv, v_new),
            "iw": equal_signal(iw, w_new),

            "AccDEY": equal_signal(AccDEY, front[:,0]),
            "AccDEZ": equal_signal(AccDEZ, front[:,1]),

            "AccNDEY": equal_signal(AccNDEY, rear[:,0]),
            "AccNDEZ": equal_signal(AccNDEZ, rear[:,1]),

            "a": equal_signal(a, housing),
            "s": equal_signal(s, speed)
        }

        if all(checks.values()):

            ok += 1
            status = "OK"
            print(f"OK: {row['Archivo_Nuevo']}")

        else:

            errors += 1
            status = "ERROR"

            failed = [k for k,v in checks.items() if not v]

            print(f"ERROR -> {row['Archivo_Nuevo']}  señales: {failed}")

        report.append({
            "archivo_nuevo": row["Archivo_Nuevo"],
            "archivo_original": row["ID_Original"],
            "status": status
        })

    except Exception as e:

        errors += 1

        print(f"FALLO leyendo {row['Archivo_Nuevo']} -> {e}")

        report.append({
            "archivo_nuevo": row["Archivo_Nuevo"],
            "archivo_original": row["ID_Original"],
            "status": "READ_ERROR"
        })


# =============================
# GUARDAR REPORTE
# =============================

report_df = pd.DataFrame(report)
report_df.to_csv("validation_report.csv", index=False)

# =============================
# RESULTADO FINAL
# =============================

print("\n========================")
print("RESULTADO FINAL")
print("========================")

print(f"Archivos verificados: {total}")
print(f"Correctos: {ok}")
print(f"Errores: {errors}")

print("\nReporte guardado en:")
print("validation_report.csv")