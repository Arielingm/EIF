# EIF – Rama `jetson`

Implementación embebida del modelo **Extended Isolation Forest (EIF)** para
la detección no supervisada de fallos electromecánicos en motores de
inducción, adaptada para ejecutarse en una **NVIDIA Jetson Nano Developer
Kit (4 GB)**.

Esta rama contiene la solución de inferencia en campo del modelo entrenado
con la fuente de características de **vibración** (la de mejor desempeño,
ver Capítulo 3 del TFM), pensada para operar de forma autónoma sobre el
dispositivo embebido, sin depender de un clúster H2O ni de una conexión a
un entorno de cómputo externo.

## Contenido

```
.
├── infer_eif_jetson.py       # Script principal de inferencia
├── modelo_vibracion.zip      # Modelo EIF exportado en formato MOJO (no versionado, ver abajo)
├── h2o-genmodel.jar          # Runtime de inferencia de H2O (no versionado, ver abajo)
├── norm_params_vibracion.npz # Parámetros de normalización (mu, sigma, threshold)
└── README.md
```

> **Nota:** `modelo_vibracion.zip` y `h2o-genmodel.jar` no se versionan en
> el repositorio por su tamaño. Ver la sección [Obtención de artefactos](#obtención-de-artefactos-del-modelo)
> para generarlos/descargarlos.

## Qué hace el script

`infer_eif_jetson.py` implementa el pipeline completo descrito en el
Capítulo 2 (Metodología) y validado en el Capítulo 3 (Resultados) del TFM:

1. Carga las cinco señales de vibración crudas (`front_DE_Y`, `front_DE_Z`,
   `rear_NDE_Y`, `rear_NDE_Z`, `housing_X`), muestreadas a `fs = 20 000 Hz`.
2. Calcula automáticamente la duración de la señal recibida y la segmenta
   en ventanas de **1 segundo** (20 000 muestras/ventana) — no asume una
   duración fija de antemano.
3. Para cada ventana y cada canal, calcula los tres descriptores usados en
   el entrenamiento: **entropía de permutación (PE)**, **dimensión fractal
   de Katz (FD)** y **curtosis** → 5 canales × 3 descriptores = 15
   características.
4. Normaliza las características con la media (`mu`) y desviación estándar
   (`sigma`) calculadas sobre los datos **sanos** de entrenamiento.
5. Ejecuta la inferencia del modelo EIF exportado en formato **MOJO**,
   invocando el entorno de ejecución Java (JRE) mediante la utilidad
   `hex.genmodel.tools.PredictCsv` mediante `h2o-genmodel.jar`. **No** se
   requiere el paquete Python de H2O ni un clúster H2O activo.
6. Compara la puntuación de anomalía de cada ventana contra el umbral
   calibrado, registra el resultado de cada ventana en un log, y emite un
   **veredicto global** (máquina sana / fallo detectado) según la fracción
   de ventanas anómalas sobre el total de la señal analizada.

## Requisitos

En la Jetson Nano (o cualquier equipo con JRE y Python 3):

- **Java** (JRE 8 o superior) — verificado sobre JetPack 4.6.x / L4T
  R32.7.4 con `openjdk-11-jre-headless`.
- **Python 3.6+** con `numpy` instalado:

  ```bash
  pip3 install numpy --break-system-packages   # o sin la bandera según el entorno
  ```

No se requieren otras dependencias de Python (ni `h2o`, ni `scikit-learn`,
ni `scipy`) — las funciones de extracción de características (PE, FD,
curtosis) están implementadas de forma nativa con `numpy` para minimizar
el consumo de recursos del dispositivo.

## Obtención de artefactos del modelo

### 1. Modelo MOJO (`modelo_vibracion.zip`)

Se exporta desde el entorno de entrenamiento (H2O, Python) tras
seleccionar el modelo EIF entrenado con la fuente de vibración:

```python
model.download_mojo(path="modelo_vibracion.zip", get_genmodel_jar=True)
```

El parámetro `get_genmodel_jar=True` descarga automáticamente
`h2o-genmodel.jar` junto con el modelo.

### 2. Parámetros de normalización (`norm_params_vibracion.npz`)

Generado una única vez a partir de las estadísticas del conjunto de
entrenamiento sano ($\mu_r$, $\sigma_r$, Sección 2.6.5 del TFM) y del
umbral de anomalía calibrado (media + 3·std, Sección 2.7 del TFM):

```python
import numpy as np

np.savez(
    "norm_params_vibracion.npz",
    mu=mu_vector,          # shape (15,)
    sigma=sigma_vector,    # shape (15,)
    threshold=threshold,   # escalar
)
```

Copia ambos archivos (`modelo_vibracion.zip`, `h2o-genmodel.jar`) junto con
`norm_params_vibracion.npz` al directorio de trabajo en la Jetson Nano.

## Uso

Cada canal de vibración se entrega como un archivo de una sola columna
(CSV o texto plano, sin cabecera) con las muestras crudas de aceleración
en *g*:

```bash
python3 infer_eif_jetson.py \
    --front-de-y front_DE_Y.csv \
    --front-de-z front_DE_Z.csv \
    --rear-nde-y rear_NDE_Y.csv \
    --rear-nde-z rear_NDE_Z.csv \
    --housing housing_X.csv \
    --mojo modelo_vibracion.zip \
    --genmodel-jar h2o-genmodel.jar \
    --norm-params norm_params_vibracion.npz \
    --log-file inferencia.log
```

### Salida

El script escribe en `inferencia.log` (y en consola) el resultado de cada
ventana procesada:

```
2026-09-09 10:15:03 [INFO] Iniciando inferencia embebida EIF (fuente: vibracion)
2026-09-09 10:15:03 [INFO] Senal cargada: 600.0 s por canal, fs=20000 Hz
2026-09-09 10:15:03 [INFO] Senal segmentada en 600 ventanas de 1 s
2026-09-09 10:15:04 [INFO] Extrayendo caracteristicas (PE, FD de Katz, curtosis)...
2026-09-09 10:15:41 [INFO] Caracteristicas normalizadas. Umbral de anomalia: 0.4375
2026-09-09 10:15:41 [INFO] Ejecutando inferencia MOJO via JRE (hex.genmodel.tools.PredictCsv)...
2026-09-09 10:15:44 [INFO] Ventana 0000 [     0s-     1s] score=0.3912 -> sano
2026-09-09 10:15:44 [INFO] Ventana 0001 [     1s-     2s] score=0.4021 -> sano
...
2026-09-09 10:15:47 [INFO] ============================================================
2026-09-09 10:15:47 [INFO] Resumen: 12/600 ventanas anomalas (2.0%) sobre 10.0 min de senal
2026-09-09 10:15:47 [INFO] Veredicto global: MAQUINA SANA
2026-09-09 10:15:47 [INFO] ============================================================
```

El criterio de veredicto global (fracción de ventanas anómalas necesarias
para declarar fallo) se controla mediante la constante
`FAULT_FRACTION_THRESHOLD` al inicio del script (por defecto, 20 %).

## Referencias

- Metodología completa del pipeline de características y del modelo EIF:
  Capítulo 2 del TFM.
- Resultados de desempeño por fuente de características y validación del
  sistema embebido: Capítulo 3 del TFM.
- Trabajo de referencia (metodología EIF original): González et al.,
  *"Detection of wind turbine rotor imbalance using unsupervised
  output-only vibration data analysis"*, Energy and AI, 2025.
  [https://doi.org/10.1016/j.egyai.2025.100565](https://doi.org/10.1016/j.egyai.2025.100565)
