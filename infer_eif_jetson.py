"""
infer_eif_jetson.py

Script de inferencia embebida para el sistema de deteccion de anomalias
en el motor de induccion, ejecutado sobre NVIDIA Jetson Nano.

Pipeline:
    1. Carga la senal de vibracion cruda (5 canales: front_DE_Y, front_DE_Z,
       rear_NDE_Y, rear_NDE_Z, housing_X), muestreada a fs=20000 Hz.
    2. Segmenta la senal en ventanas de 1 s (O=20000 muestras/ventana). El
       numero de ventanas se calcula automaticamente a partir de la
       duracion real de la senal recibida, sin asumir una duracion fija.
    3. Para cada ventana y cada canal, calcula los 3 descriptores
       estadisticos usados en el pipeline de entrenamiento: entropia de
       permutacion (PE), dimension fractal de Katz (FD) y curtosis.
       Con 5 canales x 3 descriptores = 15 caracteristicas (fuente
       "vibracion", la de mejor desempeno segun el Capitulo 3 del TFM).
    4. Normaliza las caracteristicas de cada ventana con la media y
       desviacion estandar calculadas sobre los datos sanos de
       entrenamiento (almacenadas en norm_params.npz).
    5. Ejecuta la inferencia del modelo EIF exportado en formato MOJO,
       invocando el entorno de ejecucion Java (JRE) mediante el jar
       h2o-genmodel.jar (herramienta hex.genmodel.tools.PredictCsv). No
       se requiere un cluster H2O completo ni el paquete h2o de Python.
    6. Compara la puntuacion de anomalia de cada ventana contra el umbral
       calibrado, registra el resultado de cada ventana en el log, y
       emite un veredicto global (sano / anomalo) para toda la senal de
       entrada segun la fraccion de ventanas que superan el umbral.

Uso:
    python3 infer_eif_jetson.py \
        --front-de-y front_DE_Y.csv --front-de-z front_DE_Z.csv \
        --rear-nde-y rear_NDE_Y.csv --rear-nde-z rear_NDE_Z.csv \
        --housing housing_X.csv \
        --mojo modelo_vibracion.zip \
        --genmodel-jar h2o-genmodel.jar \
        --norm-params norm_params_vibracion.npz \
        --log-file inferencia.log

Cada archivo de canal es un CSV/texto de una sola columna con las
muestras crudas de aceleracion (g), sin cabecera, muestreadas a fs=20000 Hz.

norm_params_vibracion.npz debe contener tres arreglos:
    mu        -> vector de medias (15,), del conjunto de entrenamiento sano
    sigma     -> vector de desviaciones estandar (15,), idem
    threshold -> escalar, umbral de anomalia calibrado (media + 3*std)
"""

import argparse
import csv
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuracion fija del pipeline (debe coincidir con la usada en entrenamiento)
# ---------------------------------------------------------------------------
FS = 20_000          # Hz, frecuencia de muestreo
WINDOW_SECONDS = 1    # duracion de cada ventana de analisis
O = FS * WINDOW_SECONDS  # muestras por ventana (20000)

# Parametros de entropia de permutacion (Bandt & Pompe), segun Seccion 2.6.1
PE_ORDER = 6
PE_DELAY = 1

# Orden de los canales de vibracion tal como se entrena el modelo (fuente
# "vibracion", 5 canales x 3 descriptores = 15 caracteristicas)
CHANNEL_NAMES = [
    "front_DE_Y", "front_DE_Z",
    "rear_NDE_Y", "rear_NDE_Z",
    "housing_X",
]
DESCRIPTOR_NAMES = ["PE", "FD", "Kurt"]

# Fraccion minima de ventanas anomalas dentro de la senal completa para
# emitir un veredicto global de fallo. Ajustable segun tolerancia deseada.
FAULT_FRACTION_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Extraccion de caracteristicas (PE, FD de Katz, curtosis)
# ---------------------------------------------------------------------------
def permutation_entropy(x: np.ndarray, order: int = PE_ORDER, delay: int = PE_DELAY) -> float:
    """Entropia de permutacion de Bandt & Pompe (Ecuacion PE, Seccion 2.6.1)."""
    n = len(x)
    permutations = {}
    for i in range(n - (order - 1) * delay):
        window = x[i : i + order * delay : delay]
        pattern = tuple(np.argsort(window))
        permutations[pattern] = permutations.get(pattern, 0) + 1

    counts = np.array(list(permutations.values()), dtype=float)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def katz_fractal_dimension(x: np.ndarray) -> float:
    """Dimension fractal de Katz (Ecuacion FD, Seccion 2.6.2)."""
    n = len(x)
    dists = np.sqrt(1.0 + np.diff(x) ** 2)
    length = float(np.sum(dists))
    diag = np.sqrt(np.arange(n) ** 2 + (x - x[0]) ** 2)
    max_dist = float(np.max(diag))
    if max_dist == 0 or length == 0:
        return 0.0
    return float(np.log10(n) / (np.log10(max_dist / length) + np.log10(n)))


def kurtosis(x: np.ndarray) -> float:
    """Curtosis (Ecuacion Kurt, Seccion 2.6.3), definicion no centrada en 0."""
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 4))


def extract_window_features(window_per_channel: dict) -> np.ndarray:
    """
    Calcula PE, FD y curtosis para cada canal de una ventana de 1 s, en el
    mismo orden de columnas usado durante el entrenamiento del modelo:
    [PE_ch1, FD_ch1, Kurt_ch1, PE_ch2, FD_ch2, Kurt_ch2, ...]
    """
    features = []
    for ch in CHANNEL_NAMES:
        x = window_per_channel[ch]
        features.append(permutation_entropy(x))
        features.append(katz_fractal_dimension(x))
        features.append(kurtosis(x))
    return np.array(features, dtype=float)


# ---------------------------------------------------------------------------
# Carga de senales y segmentacion en ventanas
# ---------------------------------------------------------------------------
def load_channel(path: Path) -> np.ndarray:
    """Carga un canal de vibracion desde un archivo de una columna."""
    return np.loadtxt(path, dtype=float)


def segment_into_windows(signals: dict) -> list:
    """
    Segmenta las senales cargadas en ventanas de O muestras (1 s). El
    numero de ventanas se determina a partir de la longitud real de la
    senal, sin asumir una duracion fija de antemano.
    """
    lengths = {ch: len(sig) for ch, sig in signals.items()}
    min_len = min(lengths.values())
    n_windows = min_len // O

    if n_windows == 0:
        raise ValueError(
            f"La senal recibida ({min_len} muestras, {min_len / FS:.2f} s) "
            f"es mas corta que una ventana de {WINDOW_SECONDS} s."
        )

    windows = []
    for j in range(n_windows):
        start, end = j * O, (j + 1) * O
        windows.append({ch: signals[ch][start:end] for ch in CHANNEL_NAMES})
    return windows


# ---------------------------------------------------------------------------
# Inferencia del modelo EIF (MOJO) via h2o-genmodel.jar
# ---------------------------------------------------------------------------
def run_mojo_batch(
    feature_matrix: np.ndarray,
    mojo_path: Path,
    genmodel_jar: Path,
    work_dir: Path,
) -> np.ndarray:
    """
    Ejecuta la inferencia de todas las ventanas en un solo lote, invocando
    la utilidad hex.genmodel.tools.PredictCsv del JRE local. Evita
    levantar un cluster H2O y solo requiere el jar h2o-genmodel.jar junto
    con el modelo exportado en formato MOJO.
    """
    input_csv = work_dir / "features_input.csv"
    output_csv = work_dir / "features_output.csv"

    header = [f"{d}_{ch}" for ch in CHANNEL_NAMES for d in DESCRIPTOR_NAMES]
    with open(input_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(feature_matrix.tolist())

    cmd = [
        "java", "-cp", str(genmodel_jar),
        "hex.genmodel.tools.PredictCsv",
        "--mojo", str(mojo_path),
        "--input", str(input_csv),
        "--output", str(output_csv),
        "--decimal",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Fallo la inferencia MOJO (codigo {result.returncode}).\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    with open(output_csv, newline="") as f:
        reader = csv.DictReader(f)
        score_col = next(
            (c for c in reader.fieldnames if "predict" in c.lower() or "score" in c.lower()),
            reader.fieldnames[0],
        )
        scores = [float(row[score_col]) for row in reader]

    return np.array(scores, dtype=float)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inferencia embebida del modelo EIF (fuente vibracion) en Jetson Nano."
    )
    p.add_argument("--front-de-y", required=True, type=Path)
    p.add_argument("--front-de-z", required=True, type=Path)
    p.add_argument("--rear-nde-y", required=True, type=Path)
    p.add_argument("--rear-nde-z", required=True, type=Path)
    p.add_argument("--housing", required=True, type=Path)
    p.add_argument("--mojo", required=True, type=Path, help="Modelo EIF exportado en formato MOJO (.zip)")
    p.add_argument("--genmodel-jar", required=True, type=Path, help="Ruta a h2o-genmodel.jar")
    p.add_argument("--norm-params", required=True, type=Path, help="Archivo .npz con mu, sigma y threshold")
    p.add_argument("--log-file", default="inferencia.log", type=Path)
    return p.parse_args()


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("eif_jetson")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.log_file)

    logger.info("Iniciando inferencia embebida EIF (fuente: vibracion)")

    # 1. Cargar senales crudas
    signals = {
        "front_DE_Y": load_channel(args.front_de_y),
        "front_DE_Z": load_channel(args.front_de_z),
        "rear_NDE_Y": load_channel(args.rear_nde_y),
        "rear_NDE_Z": load_channel(args.rear_nde_z),
        "housing_X": load_channel(args.housing),
    }
    duration_s = min(len(s) for s in signals.values()) / FS
    logger.info(f"Senal cargada: {duration_s:.1f} s por canal, fs={FS} Hz")

    # 2. Segmentar en ventanas de 1 s
    windows = segment_into_windows(signals)
    logger.info(f"Senal segmentada en {len(windows)} ventanas de {WINDOW_SECONDS} s")

    # 3. Extraer caracteristicas (PE, FD, Kurt) por ventana y canal
    logger.info("Extrayendo caracteristicas (PE, FD de Katz, curtosis)...")
    feature_matrix = np.array([extract_window_features(w) for w in windows])

    # 4. Normalizar con mu, sigma del conjunto de entrenamiento sano
    norm = np.load(args.norm_params)
    mu, sigma, threshold = norm["mu"], norm["sigma"], float(norm["threshold"])
    feature_matrix_norm = (feature_matrix - mu) / sigma
    logger.info(f"Caracteristicas normalizadas. Umbral de anomalia: {threshold:.4f}")

    # 5. Inferencia del modelo EIF (MOJO) para todas las ventanas
    with tempfile.TemporaryDirectory() as tmp:
        logger.info("Ejecutando inferencia MOJO via JRE (hex.genmodel.tools.PredictCsv)...")
        scores = run_mojo_batch(
            feature_matrix_norm, args.mojo, args.genmodel_jar, Path(tmp)
        )

    # 6. Log por ventana y veredicto global
    n_anomalous = 0
    for j, score in enumerate(scores):
        is_anomalous = score > threshold
        n_anomalous += int(is_anomalous)
        t_start = j * WINDOW_SECONDS
        logger.info(
            f"Ventana {j:04d} [{t_start:6.0f}s-{t_start + WINDOW_SECONDS:6.0f}s] "
            f"score={score:.4f} -> {'ANOMALO' if is_anomalous else 'sano'}"
        )

    fraction_anomalous = n_anomalous / len(scores)
    verdict = "FALLO DETECTADO" if fraction_anomalous > FAULT_FRACTION_THRESHOLD else "MAQUINA SANA"

    logger.info("=" * 60)
    logger.info(
        f"Resumen: {n_anomalous}/{len(scores)} ventanas anomalas "
        f"({fraction_anomalous:.1%}) sobre {duration_s / 60:.1f} min de senal"
    )
    logger.info(f"Veredicto global: {verdict}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
