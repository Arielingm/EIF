# Flujo completo: de los tipos de fallo al modelo validado

Este documento explica el flujo de principio a fin, en el orden en que ocurre realmente:
qué fallos hay → cómo se convierten en features → cómo se entrena el modelo → cómo se valida.
Complementa a `METODOLOGIA.md` (que describe la arquitectura del código); aquí el foco es el
*flujo de datos*, paso a paso.

---

## 1. Tipos de fallo del banco de ensayos

El banco de ensayos (`Init_Singal/Explicacion.pdf`) contiene un motor de inducción de 4 polos al
que se le provocan físicamente distintos fallos, intercambiando la máquina de ensayo. Cada
máquina se identifica con un código de una letra (columna `Maquina` del índice):

| Código | Fallo físico | Familia |
|---|---|---|
| `h` | Sano (rotor y rodamiento sin defecto) | — (referencia) |
| `e` | Máquina sana acoplada con una barra rota (incipiente) | barra rota |
| `b` | Una barra del rotor rota | barra rota |
| `v` | Varias barras del rotor rotas | barra rota |
| `p` | Barra rota en estado de "pronóstico" (severidad intermedia) | barra rota |
| `g` | Rodamiento con daño tipo ranura (*groove*) | rodamiento |
| `o` | Rodamiento con agujero de 6mm | rodamiento |
| `r` | Rodamiento con agujero de 3mm | rodamiento |
| `c` | Rodamiento dañado "real" (desgaste, no defecto artificial) | rodamiento |

Estas dos familias (`barra_rota` = `{e,b,v,p}`, `rodamiento` = `{g,o,r,c}`) son las que se usan
para reportar métricas por separado en todos los notebooks (`utils` no las centraliza porque cada
notebook las redeclara localmente junto a su lógica de evaluación, pero son siempre las mismas).

Cada fallo, además, se ensaya bajo distintas condiciones de operación, también registradas en el
índice:

- **Control** (`Control`): `l` = conexión directa a red, `s` = control escalar en variador,
  `d` = control vectorial DTC en variador. Solo `s` y `d` se usan (ver §1.1).
- **Velocidad** (`Velocidad`): consigna de velocidad mecánica — `S1500`, `S1200`, `S900` rpm
  (regímenes estables usados), más `S20`, `STrap`, `SSteps`, `SStart` (excluidos, ver §1.1).
- **Carga** (`Carga`): nivel de par resistivo aplicado por el motor/generador de carga —
  `L0` (0%) a `L100` (100%), y variantes escalonadas/variables.

Combinando fallo + control + velocidad se construye la etiqueta legible `Fallo`
(p. ej. `fallo_barra_rota_DTC_S1500`, `fallo_rodamiento_groove_scalar_S900`), y así es como se
agrupan los experimentos en `features_fft/test/<Fallo>.csv`.

### 1.1 Filtrado: qué experimentos se descartan y por qué

No todos los experimentos del banco entran en el pipeline. `scripts/prepare_index.py` marca como
**excluidos** (`Split="excluido"` en el índice) los que no son relevantes:

1. **Regímenes transitorios y `S20`**: solo se usan `S1500`, `S1200`, `S900` (velocidades
   estables). Se excluyen `STrap` (rampa), `SSteps` (escalonado), `SStart` (arranque) — el
   fenómeno físico durante un transitorio no es comparable a régimen estable — y `S20` (20 rpm,
   demasiado baja para ser representativa).
2. **Control `l` (grid directo)**: hay muy pocos experimentos sanos con conexión directa a red
   (representación insuficiente para que el modelo aprenda qué es "normal" en ese régimen).

De 217 experimentos totales del banco, **146 son relevantes** (71 excluidos: 45 por transitorio,
22 por control grid directo, 4 por ambos motivos). De esos 146: 27 son sanos, 119 son fallos.

### 1.2 Split train / val / test

Solo con los 27 sanos relevantes se hace un split 70/15/15 (semilla fija = 42):

- **train**: 18 experimentos sanos — con esto y solo esto se entrena cada modelo.
- **val**: 4 experimentos sanos — se usa para fijar el umbral de anomalía final (§4.2) y, en el
  caso del Transformer, también para early stopping durante el entrenamiento.
- **test**: 5 experimentos sanos + **todos** los 119 experimentos de fallo.

Ningún fallo se usa nunca para entrenar ni para fijar el umbral — solo aparecen en la evaluación
final. Esto es lo que hace que la detección sea "no supervisada": el modelo nunca ve un ejemplo de
fallo hasta que se le pide predecir sobre él.

---

## 2. Extracción de características

Cada archivo `.mat` de un experimento contiene ~100 segundos de señal a 20 kHz: 3 corrientes de
fase (`u, v, w`) y 5 canales de vibración (`front_DE_Y/Z`, `rear_NDE_Y/Z`, `housing`).

### 2.1 Ventaneo

La señal completa se corta en **ventanas de 1 segundo** (20 000 muestras). Un experimento típico
(100s) produce ~100 ventanas. Cada ventana es la unidad mínima que se convierte en un vector de
features.

### 2.2 Un extractor distinto por tipo de señal

Por cada ventana, `utils/extraction.py` (función `procesar_archivo`) aplica:

- **Eléctrica (`u,v,w`) → `espectro_db`**: FFT con ventana de Hanning, normalizada y en dB,
  recortada a 200 Hz (201 bins por señal). Motivo: en MCSA (Motor Current Signature Analysis)
  clásico, los fallos de barra rota se manifiestan como bandas laterales alrededor de la
  frecuencia de red, visibles directamente en el espectro FFT.
- **Vibración (5 canales) → `envolvente_espectro_db`**: se extrae la envolvente de amplitud
  (transformada de Hilbert), se le resta la media, y se calcula su FFT en dB, recortada a 1000 Hz
  (1001 bins por señal). Motivo: los fallos de rodamiento generan impactos modulados en amplitud
  sobre una portadora de alta frecuencia, que el FFT directo no revela bien; el envelope spectrum
  es la técnica de referencia en la literatura para demodular esa señal (justificación completa
  con referencias en `01_baseline_armonicos.ipynb`, sección 3).

Cada ventana termina como un vector de **5608 features**: 3×201 (eléctrica) + 5×1001 (vibración).

### 2.3 Estandarización (z-score)

La media y desviación típica de cada una de las 5608 columnas se calculan **solo con las ventanas
de train** (`features_fft/medias_sanos.csv`). Val y test se estandarizan con esos mismos
estadísticos — nunca se recalculan sobre datos que el modelo no debería "haber visto todavía",
para no filtrar información del futuro hacia el entrenamiento.

### 2.4 Resultado en disco

```
features_fft/
├── medias_sanos.csv        # media y std por columna (calculadas solo con train)
├── train/sano_train.csv    # 18 experimentos × 100 ventanas = 1800 filas × 5608 columnas
├── val/sano.csv             # 4 experimentos × 100 ventanas = 400 filas
└── test/
    ├── sano.csv                          # 5 experimentos × 100 ventanas = 500 filas
    ├── fallo_barra_rota_DTC_S1500.csv     # una fila por ventana, agrupado por (fallo, control, velocidad)
    └── ... (24 grupos de fallo)
```

Se genera con `python scripts/extract_features.py` y se valida con
`python scripts/validate_features.py` (comprueba estructura, ausencia de NaN/infinitos,
estandarización correcta en train, y consistencia del nº de ventanas contra el índice).

---

## 3. Entrenamiento del modelo

Los tres modelos de ML (Isolation Forest, Extended Isolation Forest, Transformer autoencoder)
comparten el mismo flujo de entrenamiento, solo cambia el algoritmo:

### 3.1 Datos de entrada: solo sanos de train

El modelo **nunca ve un fallo durante el entrenamiento**. Se entrena únicamente con
`features_fft/train/sano_train.csv` (1800 ventanas de 18 experimentos sanos). Esto es lo que
convierte el problema en detección de anomalías no supervisada: el modelo aprende "cómo es una
ventana normal", no "cómo distinguir sano de fallo".

Cada uno de los tres modelos se entrena, además, **tres veces** — una por variante de fuente de
señal — seleccionando solo el subconjunto de columnas correspondiente del mismo CSV:

- **eléctrica**: columnas `u_bin*, v_bin*, w_bin*` (603 features).
- **vibración**: columnas de los 5 canales de vibración (5005 features).
- **híbrida**: las 5608 columnas completas.

### 3.2 Búsqueda de hiperparámetros (Optuna)

Para cada variante, Optuna prueba distintas combinaciones de hiperparámetros del modelo y elige la
mejor según un criterio de **compacidad**: se entrena con esos hiperparámetros sobre train, se
calcula el score de anomalía de cada ventana de train, se agregan por experimento (mediana — ver
§3.3) y se mide `std / rango` de esas 18 medianas. Cuanto más compacta (menor `std/rango`) sea la
distribución de scores de los sanos, mejor separación cabe esperar frente a los fallos más
adelante.

Hiperparámetros buscados por modelo:

| Modelo | Hiperparámetros | Rango |
|---|---|---|
| Isolation Forest | `n_estimators`, `max_samples` | 50-300 (paso 25), 32-256 (paso 32) |
| Extended Isolation Forest | `ntrees`, `sample_size`, `extension_level` | 30-150, 32-128 (paso 32), 0-`min(50, n_features-1)` |
| Transformer autoencoder | `patch_size`, `d_model`, `num_layers`, `lr` | {32,64,128}, {32,64,128}, 1-3, 1e-4 a 5e-3 (log) — aquí Optuna minimiza el *loss* de reconstrucción en val, no la compacidad, porque es el criterio estándar de selección de modelo para autoencoders |

`extension_level` en EIF está acotado a 50 (en vez de `n_features-1`) porque, medido
empíricamente, el coste de entrenar un árbol de EIF en H2O crece muy rápido con ese hiperparámetro
en dimensión alta — sin el límite, un solo trial llegó a tardar más de 30 minutos con 5608
features.

### 3.3 Agregación por experimento

Cada experimento sano de train genera 100 scores de ventana. Para todas las decisiones de
compacidad/umbral/evaluación, esos 100 scores se reducen a **un solo número por experimento**
tomando la **mediana** (`utils.eval.agregar_por_experimento`). La razón: la pregunta que importa
("¿este experimento es sano o es un fallo?") es por experimento, no por ventana individual, y la
mediana es robusta a que alguna ventana puntual tenga un score raro por ruido.

### 3.4 Modelo final

Con los mejores hiperparámetros encontrados por Optuna, se reentrena el modelo una última vez
sobre train — este es el modelo que se usa en la validación.

---

## 4. Validación del modelo

### 4.1 ¿Qué se evalúa?

El modelo final se aplica a:

- **train** y **val** (sanos, para fijar el umbral — ver §4.2).
- **cada uno de los 24 grupos de fallo** de test, más **test/sano.csv** (5 experimentos sanos que
  el modelo tampoco vio nunca, ni para entrenar ni para fijar el umbral — sirven de control).

En cada caso, los scores de ventana se agregan por experimento con la mediana (§3.3), igual que en
entrenamiento.

### 4.2 Umbral de anomalía

Con los scores de **train + val combinados** (22 experimentos sanos en total: 18+4), se calcula:

```
umbral = media_sanos + 3 · std_sanos
```

Asumiendo que la distribución de scores de los sanos es aproximadamente normal, ese umbral cubre
~99.7% de esa distribución — todo lo que quede por encima se clasifica como anómalo. Se combinan
train y val (en vez de usar solo train) para tener una estimación más robusta de "cómo puntúa un
sano", con más datos.

*(Nota: con un umbral fijo así, y una muestra de solo 22 sanos, hay riesgo de que un experimento
sano con score inusualmente alto en val infle mucho el `std` y deje el umbral demasiado
conservador — esto se documentó como limitación real observada en el notebook 02, donde
`precision`/`recall` salieron en 0 pese a que el AUC mostraba separación por encima del azar.)*

### 4.3 Métricas

Con el umbral fijado, cada experimento de test (fallo o sano) se clasifica como
`score_mediana > umbral → anómalo`. Se calculan (`utils.eval.calcular_metricas`, basado en
scikit-learn):

- **Precision, recall, F1**: dependen del umbral concreto.
- **AUC** (area bajo la curva ROC): no depende del umbral — mide si el modelo *ordena* mejor los
  fallos que los sanos, independientemente de dónde se ponga la línea de corte. Es la métrica más
  informativa cuando el umbral es incierto (como aquí, por el punto anterior).
- **Matriz de confusión** (tp/fp/tn/fn).

Estas métricas se calculan **tres veces** por variante de fuente de señal: una comparando
`sano` vs. familia `barra_rota`, otra `sano` vs. familia `rodamiento`, y una global (`sano` vs.
todos los fallos juntos) — para poder ver si el modelo detecta mejor un tipo de fallo que otro.

### 4.4 Resultado

Todo el resultado de una combinación (modelo, fuente de señal) — hiperparámetros elegidos, umbral,
métricas por familia y global — se guarda en `resultados/<modelo>_<fuente>.json`
(`utils.eval.guardar_resultado`). El notebook `05_comparacion_resultados.ipynb` carga todos esos
JSON y construye las tablas/gráficas comparativas finales, entre modelos y entre fuentes de señal.

---

## Resumen visual del flujo

```
Motor_DB/data/*.mat (por tipo de fallo, control, velocidad, carga)
        │
        ▼  scripts/prepare_index.py
Motor_DB/index/master_index.csv   (Split: train / val / test / excluido)
        │
        ▼  scripts/extract_features.py  (ventaneo 1s + espectro_db / envolvente_espectro_db + z-score)
features_fft/{train,val,test}/*.csv
        │
        ▼  notebooks 02/03/04, por cada fuente (eléctrica / vibración / híbrida):
        │     1. Optuna busca hiperparámetros minimizando compacidad de sanos-train
        │     2. Reentrena modelo final con los mejores hiperparámetros (solo train)
        │     3. Umbral = media+3·std de sanos (train+val)
        │     4. Evalúa sobre val + cada grupo de fallo de test → métricas
        ▼
resultados/<modelo>_<fuente>.json
        │
        ▼  05_comparacion_resultados.ipynb
Tablas y gráficas comparativas finales
```
