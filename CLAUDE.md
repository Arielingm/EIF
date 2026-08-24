# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

TFM sobre detección de anomalías en motores de inducción mediante señales eléctricas y de vibración, usando **Extended Isolation Forest (EIF)** (H2O.ai) sobre espectros FFT de las señales. El pipeline completo es:

`Motor_DB/data/*.mat` (HDF5 v7.3) → `scripts/extract_features.py` (FFT → dB → z-score) → `features_fft/{train,val,test}/*.csv` → notebooks `FFT_train_*.ipynb` (Optuna + H2O EIF) → `resultados_fft_*/` (modelo, umbral, gráficas).

No hay build/lint/test tradicional: es un proyecto de investigación en notebooks + scripts de procesamiento por lotes que se ejecutan directamente.

## Entorno y comandos

- Python 3.14, entorno virtual en `ambiente/` (creado con `python3 -m venv ambiente`, no está en git). Activar con `source ambiente/bin/activate.fish` (o `.sh`/`.csv` según shell).
- **`requirements.txt` está vacío** — las dependencias reales están instaladas en `ambiente/` pero no declaradas. Antes de reinstalar el entorno, generar el `requirements.txt` a partir de `ambiente/lib/python3.14/site-packages` (paquetes clave: `h2o`, `optuna`, `optuna-dashboard`, `h5py`, `pandas`, `numpy`, `seaborn`, `plotly`).
- Ejecutar scripts de procesamiento como módulos desde la raíz del repo (usan rutas relativas tipo `Motor_DB/index/master_index.csv`, `features_fft/...`):
  ```
  python scripts/extract_features.py      # extracción de features FFT (train/val/test)
  python scripts/prepare_index.py         # (re)genera Split y Fallo en master_index.csv
  python scripts/validate_features.py     # valida estructura/NaNs/estandarización de features_fft/
  python scripts/check_windows.py         # ventanas por experimento vs. index
  python analis_distribucion.py           # desglose de experimentos por split/fallo/condición
  python diagnostico_sanos_anomalos.py    # identifica sanos train/val con score anómalo (requiere H2O + Optuna study ya entrenados)
  ```
- Los notebooks (`FFT_train_eif*.ipynb`, `FFT_train_hibrido.ipynb`, `train_*.ipynb`) se ejecutan con Jupyter e inicializan H2O localmente (`h2o.init(...)`). Requieren memoria considerable (`max_mem_size` 10–16G) por el tamaño de los frames (hasta 5608–8008 features × cientos de ventanas).
- No hay suite de tests. La "validación" del pipeline de datos es `scripts/validate_features.py` (comprueba columnas, NaNs/infinitos, media≈0/std≈1 en train, y separación sanos/fallos).

## Arquitectura de datos

### Índice maestro (`Motor_DB/index/master_index.csv`)
Fuente de verdad de qué experimento es qué. Columnas clave: `Archivo_Nuevo` (nombre del `.mat` en `Motor_DB/data/`), `Maquina` (`h`=sano, resto=código de fallo, ver `MACHINE_LABELS` en `scripts/prepare_index.py`), `Control` (`l`=grid directo, `s`=scalar, `d`=DTC), `Velocidad` (`S1500`/`S1200`/`S900`/`S20`=regímenes estables; `STrap`/`SSteps`/`SStart`=transitorios), `Carga`, `Fallo` (etiqueta compuesta), `Split` (`train`/`val`/`test`).

`scripts/prepare_index.py` es el único script que **escribe** sobre este índice: asigna `Fallo` y reparte los sanos 70/15/15 en train/val/test (seed 42); todos los fallos van a `test`. Se ejecuta una vez para regenerar el índice, no en cada pipeline run.

Convención de filtrado usada consistentemente en `extract_features.py`, notebooks y scripts de análisis: solo regímenes estables (`{S1500, S1200, S900, S20}`) y solo control `d`/`s` (se excluye `l`/grid directo por representación insuficiente en train).

### Archivos `.mat` (`Motor_DB/data/`)
HDF5 v7.3, cargados con `h5py` (no `scipy.io.loadmat`, que no soporta v7.3). Estructura interna fija: `test/signals/electrical/{u,v,w}`, `test/signals/vibration/{front_DE, rear_NDE, housing_uniaxial}` (front/rear son biaxiales → columnas Y/Z), `test/signals/speed_raw`. Frecuencia de muestreo 20 kHz; cada archivo tiene 2 002 000 muestras (~100 s).

### Extracción de features (`scripts/extract_features.py`)
Ventanas de 1 s (20000 muestras) → Hanning → FFT → magnitud normalizada en dB, recortada por tipo de señal:
- Eléctricas (`u,v,w`): hasta 200 Hz (armónicos de red) → 201 bins c/u.
- Vibración (5 señales): hasta 1000 Hz (rodamiento/barra rota) → 1001 bins c/u.
- Total: 3×201 + 5×1001 = **5608 features/ventana**.

Estandarización z-score: las medias/std se calculan **solo sobre train (sanos)** y se guardan en `features_fft/medias_sanos.csv`; val/test se normalizan reusando esos estadísticos (nunca se recalculan sobre val/test — evita fuga de información).

Existen variantes de este pipeline con distinto alcance de features: `features_1/` (features estadísticos/no lineales: Katz, entropía de permutación, curtosis, RMS, pico, cresta — ver `NOMBRES_COLS` en `scripts/validate_features.py`, 48 features) vs. `features_fft/` (espectro dB, 5608 features, el enfoque actual). Los notebooks `FFT_train_*` usan `features_fft/`; `train_model_hibrido.ipynb`/`train_eif.ipynb` usan el enfoque de `features_1`.

### Modelado (notebooks)
Extended Isolation Forest vía `H2OExtendedIsolationForestEstimator`, entrenado **solo con datos sanos**. Optuna busca hiperparámetros (`ntrees`, `sample_size`, `extension_level`, `percentil_umbral`) minimizando la compactación de la distribución de scores de los sanos (`std/rango`); el umbral de anomalía se fija como percentil de esa distribución sobre sanos, no `media + k·std`. Los estudios de Optuna persisten en SQLite (`resultados_*/optuna_eif*.db`) con `load_if_exists=True`, así que relanzar una celda continúa el estudio en vez de reiniciarlo.

Agregación por experimento: cada `.mat` genera ~100 ventanas/scores; se agregan con la **mediana** del bloque (`agregar_por_experimento`/`agregar_variable` en los notebooks) antes de comparar contra el umbral, porque la decisión de "sano/fallo" es por experimento, no por ventana.

`FFT_train_hibrido.ipynb` / `train_model_hibrido.ipynb` entrenan dos modelos EIF separados (eléctrico y vibración) en vez de uno combinado — ver `resultados_hibrido*/` (guarda `optuna_elec.db` y `optuna_vib.db` por separado).

### Convención de versionado de resultados
Cada iteración del pipeline FFT vuelca a una carpeta `resultados_fft_vN(.M)/` propia (modelo, `mejores_hiperparametros.json`, `umbral.json`, `optuna_eif_fft.db`, boxplot, CSVs de resultados). Al crear una nueva variante de preprocesado/modelo, seguir esta convención en vez de sobrescribir resultados previos.

## Notas al modificar el pipeline

- Si se cambia `N_BINS_ELEC`/`N_BINS_VIB`/`F_MAX_*` en `extract_features.py`, hay que regenerar `features_fft/` completo y volver a calcular `medias_sanos.csv` — el resto del pipeline (notebooks, `validate_features.py`) asume las dimensiones actuales (5608 features) de forma implícita en varios sitios (p. ej. `EXT_MAX = N_FEATURES - 1` para el `extension_level` de EIF).
- `Init_Singal/validaciones_varias/` contiene scripts de validación de la base de datos original (duplicados, shapes, comparación contra `DataBase_original/` que no está en el repo) — son de una etapa anterior de preparación de datos, no del pipeline de features/modelado activo.
- `ambiente/` y `Motor_DB/` están en `.gitignore` (el entorno virtual y los `.mat` originales, respectivamente, no se versionan).
