import pandas as pd
import numpy as np
import h5py
import os

csv_path = "Motor_DB/index/master_index.csv"
processed_folder = "Motor_DB/data"

# =============================
# DIMENSIONES ESPERADAS
# =============================

EXPECTED_SHAPES = {
    "test/signals/electrical/u":          (2002000, 1),
    "test/signals/electrical/v":          (2002000, 1),
    "test/signals/electrical/w":          (2002000, 1),
    "test/signals/vibration/front_DE":    (2002000, 2),
    "test/signals/vibration/rear_NDE":    (2002000, 2),
    "test/signals/vibration/housing_uniaxial": (2002000, 1),
    "test/signals/speed_raw":             (2002000, 1),
}


# =============================
# CARGAR CSV
# =============================

df = pd.read_csv(csv_path)

total = len(df)
ok = 0
errors = 0
report = []

print(f"\nValidando dimensiones en {total} archivos...\n")

# =============================
# LOOP PRINCIPAL
# =============================

for _, row in df.iterrows():

    new_file = os.path.join(processed_folder, row["Archivo_Nuevo"])

    try:
        shape_errors = []

        with h5py.File(new_file, 'r') as f:
            for signal_path, expected_shape in EXPECTED_SHAPES.items():
                data = np.array(f[signal_path])

                # Normaliza siempre a (N, C) antes de comparar
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                elif data.shape[0] < data.shape[1]:  # está transpuesto
                    data = data.T

                if data.shape != expected_shape:
                    shape_errors.append(
                        f"{signal_path.split('/')[-1]}: "
                        f"esperado {expected_shape}, obtenido {data.shape}"
                    )

        if not shape_errors:
            ok += 1
            status = "OK"
            print(f"OK: {row['Archivo_Nuevo']}")
        else:
            errors += 1
            status = "ERROR"
            for err in shape_errors:
                print(f"  ERROR -> {row['Archivo_Nuevo']} | {err}")

        report.append({
            "archivo_nuevo":    row["Archivo_Nuevo"],
            "archivo_original": row["ID_Original"],
            "status":           status,
            "detalle":          "; ".join(shape_errors) if shape_errors else ""
        })

    except Exception as e:
        errors += 1
        print(f"FALLO leyendo {row['Archivo_Nuevo']} -> {e}")
        report.append({
            "archivo_nuevo":    row["Archivo_Nuevo"],
            "archivo_original": row["ID_Original"],
            "status":           "READ_ERROR",
            "detalle":          str(e)
        })

# =============================
# GUARDAR REPORTE
# =============================

report_df = pd.DataFrame(report)
report_df.to_csv("validation_shapes_report.csv", index=False)

# =============================
# RESULTADO FINAL
# =============================

print("\n========================")
print("RESULTADO FINAL")
print("========================")
print(f"Archivos verificados: {total}")
print(f"Correctos:            {ok}")
print(f"Errores:              {errors}")
print("\nReporte guardado en: validation_shapes_report.csv")