# Metodología del pipeline de detección de anomalías

Este documento describe, de forma exhaustiva y trazable, todo el pipeline implementado para el
TFM: desde los datos crudos hasta los cuatro modelos comparados. El objetivo es que cualquier
resultado se pueda seguir hasta su origen exacto (qué código lo generó, con qué datos, con qué
criterio).

## 1. Objetivo

Detectar fallos en una máquina de inducción (barras rotas y fallos de rodamiento) a partir de
señales eléctricas (corrientes de fase) y de vibración, comparando cuatro enfoques:

1. Un baseline clásico sin Machine Learning (análisis de armónicos/bandas laterales).
2. Isolation Forest.
3. Extended Isolation Forest (EIF).
4. Un autoencoder basado en Transformer.

Los tres modelos de ML se entrenan **exclusivamente con datos sanos** (detección de anomalías no
supervisada) y se evalúan sobre tres variantes de fuente de señal: eléctrica, vibración e híbrida.

## 2. Banco de ensayos y datos crudos

Los datos (`Motor_DB/data/*.mat`, formato HDF5 v7.3) provienen de un banco de ensayos con:

- Un motor de inducción de 4 polos (400V, 50Hz, 1474rpm nominal a plena carga — ficha técnica en
  `Init_Singal/Explicacion.pdf`), gobernado por un variador en **control vectorial DTC** (o
  control escalar, o conexión directa a red) que regula la **velocidad** en lazo cerrado.
- Un segundo motor/generador con su propio drive de par, que aplica la carga mecánica.
- Sensores: corrientes de fase `u, v, w`; vibración biaxial en `front_DE` y `rear_NDE`; vibración
  uniaxial en `housing`; velocidad mecánica medida (`speed_raw`).

Cada archivo `.mat` contiene 2 002 000 muestras a 20 kHz (~100 s) para cada señal.

**Nota importante de diseño** (documentada y verificada en `01_baseline_armonicos.ipynb`,
sección 4.1): al ser un motor con **control vectorial en lazo cerrado** y no alimentado
directamente a red (salvo el control `l`, excluido — ver más abajo), la relación clásica entre
deslizamiento mecánico y frecuencia eléctrica **no se cumple de forma fiable**. Esto se comprobó
empíricamente (13% de los experimentos analizados dan deslizamiento negativo, físicamente
imposible en un motor a red) y es la motivación principal para usar modelos que no dependan de un
modelo físico del fallo ni del régimen de control.

## 3. Índice maestro y criterios de selección

`Motor_DB/index/master_index.csv` es la fuente de verdad de qué experimento es qué y a qué
subconjunto (train/val/test/excluido) pertenece. Columnas: `Archivo_Nuevo, ID_Original, Generador,
Maquina, Control, Velocidad, Carga, Intento, Fecha, Fallo, Split, Motivo_Exclusion`.

Se genera con `scripts/prepare_index.py`, que:

1. Construye la etiqueta `Fallo` a partir de `Maquina` + `Control` + `Velocidad`
   (`utils/config.py`: `MACHINE_LABELS`, `CONTROL_LABELS`).
2. Marca como **excluidos** (`Split="excluido"`, con el motivo en `Motivo_Exclusion`) los
   experimentos que no son relevantes para la investigación:
   - **Transitorios**: solo se usan regímenes estables `S1500, S1200, S900`
     (excluye `STrap`=rampa, `SSteps`=escalonado, `SStart`=arranque, y `S20`=revolución
     demasiado baja y no representativa).
   - **Control grid directo (`l`)**: representación insuficiente en train para que el modelo
     aprenda qué es "normal" con conexión directa a red.
   - Ambos criterios están centralizados en `utils/config.py`
     (`VELOCIDADES_ESTABLES`, `CONTROL_EXCLUIDO`) — **fuente única**, no se redefinen en ningún
     otro script.
3. Sobre los experimentos **sanos relevantes** (`Maquina=="h"`), hace un split 70/15/15
   (train/val/test) con semilla fija (`utils.config.SEED=42`).
4. Todos los **fallos relevantes** van a `test` (nunca se entrena con fallos).

Resultado actual: 217 experimentos totales → **71 excluidos** (45 transitorio, 22 grid directo, 4
ambos) → 146 relevantes → sanos: 18 train / 4 val / 5 test; fallos: 119, todos en test.

## 4. Extracción de características

**Ejecutado por**: `scripts/extract_features.py` (CLI) → `python scripts/extract_features.py`.
Este script es un orquestador delgado: itera el índice, reparte el trabajo entre procesos
(`multiprocessing.Pool`) y guarda los CSV. **Toda la lógica de extracción vive en `utils/`** y se
importa, no se duplica — así el mismo código que documenta y justifica el notebook 01 es,
literal y exactamente, el que genera los datos con los que se entrenan los modelos de los
notebooks 02-04. (Antes de esta consolidación existía una versión antigua de
`scripts/extract_features.py` que definía su propio `espectro_db` y lo aplicaba también a las
señales de vibración; se corrigió para eliminar esa duplicación y la divergencia que causaba.)

### 4.1 Ventaneo

Cada experimento (100 s a 20 kHz) se divide en ventanas de 1 s (20 000 muestras,
`utils.config.VENTANA`), típicamente 100 ventanas por experimento (`VENTANAS_POR_EXP`).

### 4.2 Dos extractores, uno por tipo de señal (`utils/features.py`)

Decisión justificada con revisión bibliográfica en `01_baseline_armonicos.ipynb` (sección 3):

- **`espectro_db`** — señales eléctricas (`u, v, w`). FFT directa con ventana de Hanning,
  normalizada y expresada en dB, recortada a 200 Hz (`F_MAX_ELEC`, 201 bins/señal). Justificación:
  MCSA (Motor Current Signature Analysis) clásico — el fallo se manifiesta como
  armónicos/bandas laterales alrededor de la frecuencia de red, visibles directamente en el
  espectro FFT sin demodulación.

- **`envolvente_espectro_db`** — señales de vibración (`front_DE_Y/Z`, `rear_NDE_Y/Z`, `housing`).
  Envelope spectrum: se extrae la envolvente de amplitud vía transformada de Hilbert
  (`scipy.signal.hilbert`), se le resta la media, y se calcula su FFT en dB, recortada a 1000 Hz
  (`F_MAX_VIB`, 1001 bins/señal). Justificación: los defectos de rodamiento (BPFO/BPFI/BSF/FTF)
  generan impactos de amplitud modulada sobre una portadora de alta frecuencia (resonancia
  estructural) que el espectro FFT directo no revela; el envelope spectrum es la técnica de
  referencia en la literatura para demodular esa señal y dejar las frecuencias de fallo como picos
  de baja frecuencia. Referencias completas en el notebook 01.

Total: 3×201 (eléctrica) + 5×1001 (vibración) = **5608 features por ventana**.

### 4.3 Estandarización (z-score)

Media y desviación típica se calculan **solo sobre `train` (sanos)**
(`utils.extraction.calcular_y_guardar_estadisticos` → `features_fft/medias_sanos.csv`) y se
reutilizan para estandarizar `val` y `test` (`estandarizar`) — nunca se recalculan sobre val/test,
para evitar fuga de información.

### 4.4 Salida

```
features_fft/
├── medias_sanos.csv          # media y std por columna, calculadas solo con train
├── train/sano_train.csv      # 18 archivos × 100 ventanas = 1800 filas × 5608 columnas
├── val/sano.csv               # 4 archivos × 100 ventanas = 400 filas
└── test/
    ├── sano.csv                                    # 5 archivos × 100 ventanas = 500 filas
    ├── fallo_barra_rota_DTC_S1500.csv               # etc., una fila por (fallo, control, velocidad)
    └── ... (24 grupos de fallo en total)
```

Validado automáticamente por `scripts/validate_features.py` (estructura, NaNs/infinitos,
estandarización correcta en train, consistencia de nº de ventanas contra el índice).

## 5. Arquitectura del código compartido (`utils/`)

| Módulo | Contenido |
|---|---|
| `config.py` | Rutas, constantes de señal (`FS`, `VENTANA`, `F_MAX_*`, `N_BINS_*`), `NOMBRES_COLS*`, etiquetas del índice (`MACHINE_LABELS`, `CONTROL_LABELS`), **criterios de exclusión** (`VELOCIDADES_ESTABLES`, `CONTROL_EXCLUIDO`), semilla (`SEED=42`) y `fijar_semillas()`. |
| `data.py` | `cargar_indice`, `cargar_senales` (.mat → dict de señales), `cargar_velocidad` (speed_raw). |
| `features.py` | `espectro_db`, `envolvente_espectro_db` (ver §4.2). |
| `extraction.py` | `procesar_archivo` (ventaneo + extracción por ventana), `calcular_y_guardar_estadisticos`, `cargar_estadisticos`, `estandarizar`. Usado tanto por `scripts/extract_features.py` como, indirectamente, por el notebook 01 (reimplementa el mismo criterio para su propio análisis directo sobre las señales crudas). |
| `eval.py` | `agregar_por_experimento` / `agregar_variable` (mediana de ventanas → 1 score por experimento), `calcular_metricas` (precision/recall/f1/AUC/matriz de confusión vía scikit-learn), `guardar_resultado` (→ `resultados/*.json`). |
| `torch_model.py` | `TransformerAutoencoder` (autoencoder con parches 1D + `TransformerEncoder`), `entrenar_autoencoder` (con early stopping), `error_reconstruccion`. Usado por el notebook 04. |

Los `scripts/*.py` que viven en `scripts/` (no en la raíz) necesitan un pequeño bootstrap de
`sys.path` al principio (`sys.path.insert(0, ...)`) para poder hacer `from utils... import ...`,
porque Python no añade la raíz del repo al `sys.path` cuando se ejecuta un script como
`python scripts/archivo.py`. Todos los scripts relevantes (`prepare_index.py`,
`extract_features.py`, `validate_features.py`, `check_windows.py`) ya lo incluyen.

## 6. Notebooks

Cada notebook 01-04 es autocontenido: carga de datos → (features, si aplica) → entrenamiento/
análisis → evaluación → conclusiones. Los notebooks 02-04 comparten exactamente el mismo patrón de
evaluación (§7) para que la comparación del notebook 05 sea homogénea.

### 01 — `01_baseline_armonicos.ipynb` (sin ML)

No usa `features_fft/` — trabaja directamente sobre las señales crudas de `Motor_DB/data/` con sus
propias funciones (equivalentes conceptualmente a las de `utils/`, pero implementadas inline
porque el análisis es distinto: FFT de la señal completa de 100 s para máxima resolución en
frecuencia, no ventanas de 1 s).

1. Comparación visual sano vs. fallo (tiempo y espectro) para eléctrica y vibración, en una
   condición de referencia común (S1500, control DTC).
2. Revisión bibliográfica que justifica los dos extractores del §4.2.
3. Detección clásica por bandas laterales de barra rota, aplicada **solo a la señal eléctrica**:
   localiza la frecuencia fundamental `f1`, estima el deslizamiento comparando con `speed_raw`, y
   mide la amplitud relativa de las bandas `f1·(1±2s)`.
4. Evaluación sobre los 146 experimentos relevantes, con umbral `media_sanos + 3·std_sanos`.
5. Conclusión (con datos): el método no discrimina de forma fiable ni barra rota (AUC≈0.55) ni
   rodamiento (AUC≈0.36, peor que azar) en este banco, precisamente por el control DTC en lazo
   cerrado — motivando los notebooks 02-04.

### 02 — `02_isolation_forest.ipynb`

Isolation Forest (`sklearn.ensemble.IsolationForest`) sobre las 3 variantes de fuente de señal.
Optuna (`n_estimators`, `max_samples`) minimiza `std/rango` de las medianas de score de los sanos
de train — cuanto más compacta esa distribución, mejor separación cabe esperar. Umbral y métricas:
ver §7.

### 03 — `03_extended_isolation_forest.ipynb`

Mismo patrón que 02, pero con `H2OExtendedIsolationForestEstimator` (H2O). Hiperparámetros:
`ntrees`, `sample_size`, `extension_level` (acotado a `min(50, n_features-1)`: en H2O el coste de
construir cada árbol crece muy rápido con `extension_level` en dimensión alta — con 5608 features
sin ese límite, un solo trial de Optuna llegó a tardar más de 30 minutos; medido empíricamente,
cada trial con `extension_level≤50` tarda 3-50s según la variante).

### 04 — `04_transformers.ipynb`

Autoencoder basado en Transformer (`utils/torch_model.py`): cada ventana se trocea en parches de
tamaño fijo, se proyectan a un embedding, se procesan con `TransformerEncoder` (self-attention), y
un decoder lineal reconstruye cada parche. Entrenado solo con sanos, minimizando MSE de
reconstrucción. Optuna busca `patch_size`, `d_model`, `num_layers`, `lr` minimizando el loss de
validación (criterio estándar de selección de modelo para autoencoders). El error de
reconstrucción de una ventana nueva es su score de anomalía.

### 05 — `05_comparacion_resultados.ipynb`

Carga todos los `resultados/*.json`, compara (a) entre modelos (mejor variante de cada uno) y
(b) entre fuentes de señal dentro de cada modelo, con tablas y gráficas de barras.

## 7. Criterio de evaluación común (02, 03, 04)

Idéntico en los tres, para que el notebook 05 compare cosas comparables:

1. **Entrenamiento**: solo con `train` (sanos). Optuna ajusta hiperparámetros del modelo
   minimizando `std/rango` de las medianas de score de los sanos de train (compacidad).
2. **Umbral final**: con el modelo ya entrenado (mejores hiperparámetros), se calcula
   `media_sanos + 3·std_sanos` usando los scores de **train + val combinados** (ambos son sanos;
   combinarlos da una estimación más robusta que solo train). Asume aproximadamente normalidad de
   esa distribución.
3. **Agregación por experimento**: cada experimento genera ~100 scores (uno por ventana); se
   agregan con la **mediana** (`agregar_por_experimento` / `agregar_variable` para grupos con nº de
   ventanas no uniforme) — la decisión sano/fallo es por experimento, no por ventana.
4. **Métricas**: `calcular_metricas` (precision, recall, F1, AUC, matriz de confusión), calculadas
   por separado para las dos familias de fallo (`barra_rota`: máquinas `e,b,v,p`; `rodamiento`:
   máquinas `g,o,r,c`) y de forma global.
5. **Resultado guardado**: `resultados/<modelo>_<fuente>.json` con hiperparámetros, umbral y
   métricas — consumido por el notebook 05.

**Colores de gráficas**: convención fija en todos los notebooks — `steelblue` para grupos sanos,
`salmon` para grupos de fallo.

## 8. Estado actual (qué está ejecutado y qué no)

| Notebook | Código | Ejecutado con datos/criterio actuales |
|---|---|---|
| 01 baseline | ✅ completo | ✅ sí — AUC 0.55 (barra_rota), 0.36 (rodamiento) |
| 02 Isolation Forest | ✅ completo | ✅ sí — AUC global: eléctrica 0.59, vibración 0.69, híbrida 0.68 (ver nota de corrección de pipeline en el propio notebook: una ejecución previa usaba por error FFT directa en vez de envelope spectrum para vibración) |
| 03 EIF | ✅ completo | ⏳ no — construido, pendiente de ejecución completa (necesita re-ejecutarse también con `features_fft/` regenerado) |
| 04 Transformers | ✅ completo | ⏳ no — construido, nunca ejecutado |
| 05 Comparación | ✅ completo | ⏳ no — necesita 03 y 04 ejecutados primero |

## 9. Limitaciones conocidas (documentadas explícitamente en los notebooks)

- **Tamaño de muestra pequeño para el umbral paramétrico**: solo 18-22 experimentos sanos
  (train+val) para estimar `media`/`std`. En el notebook 02 esto llevó a que, pese a un AUC
  por encima del azar (0.59-0.69), `precision/recall` salieran en 0 porque el umbral quedó casi al
  nivel del máximo absoluto observado — no es un fallo del modelo, es inestabilidad de la
  estimación paramétrica con tan pocos datos.
- **El envelope spectrum (vibración) da peor AUC que la FFT directa en este dataset**: la revisión
  bibliográfica del notebook 01 justifica el envelope spectrum como técnica de referencia para
  fallos de rodamiento, y así se implementó en `utils/features.py`. Pero al corregir
  `scripts/extract_features.py` para que usara ese extractor consistentemente (antes tenía una
  copia propia de `espectro_db` desincronizada, aplicada por error también a vibración), el AUC de
  Isolation Forest en la variante vibración bajó de 0.81 a 0.69, e híbrida de 0.80 a 0.68. Es un
  resultado empírico real, no un error — queda documentado como posible línea de trabajo futuro
  (p. ej. afinar la banda de demodulación con spectral kurtosis/kurtogram en vez de usar la señal
  completa).
- **`extension_level` en EIF**: acotado a 50 por coste computacional en H2O con datos de alta
  dimensión (no es una limitación conceptual del método, es una decisión práctica documentada y
  medida).
- **`diagnostico_sanos_anomalos.py`** (raíz del repo) es un script de diagnóstico anterior a este
  pipeline (referencia `resultados_fft/` y el flujo H2O exploratorio original, no `utils/`); no
  forma parte del pipeline activo 01-05 y no se ha actualizado.
