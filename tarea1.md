# Tarea: Pipeline de detección de anomalías para TFM (fallos en máquina de inducción)

## Contexto

TFM sobre detección de fallos en máquinas de inducción (barras rotas, fallos de rodamiento, fallos de estator) usando señales eléctricas y de vibración procedentes de un banco de ensayos (régimen estacionario). El objetivo no es solo entrenar modelos, sino **comparar enfoques** y dejar todo el razonamiento documentado, porque los notebooks se van a reutilizar como base para redactar la memoria del TFM.

## Estructura general pedida

Un notebook `.ipynb` independiente **por cada modelo/enfoque**, autocontenido (carga de datos → preprocesado → extracción de features → entrenamiento/análisis → evaluación → conclusiones parciales), más un script final de comparación:

1. `01_baseline_armonicos.ipynb` — Versión sin modelo de IA
2. `02_isolation_forest.ipynb` — Isolation Forest tradicional
3. `03_extended_isolation_forest.ipynb` — Extended Isolation Forest (EIF)
4. `04_transformers.ipynb` — Modelo basado en Transformers
5. `05_comparacion_resultados.py` (o `.ipynb`) — Compara los resultados de los 4 anteriores

Cada notebook debe explicar en celdas markdown **qué se hace y por qué**, con nivel de detalle suficiente para poder copiar/adaptar esos párrafos directamente a la memoria del TFM.

## 1. Notebook 01 — Baseline sin IA (análisis de armónicos)

- Cargar señal eléctrica sana y señal eléctrica con cada tipo de fallo.
- Graficar la señal en el tiempo y su espectro (FFT / armónicos) para sano vs. cada fallo, para poder ver a simple vista si hay diferencias visibles.
- Aplicar el análisis clásico de detección por armónicos característicos (sin ningún modelo de ML) y evaluar hasta qué punto permite discriminar sano/fallo.
- Documentar conclusión: en qué casos este análisis funciona y en cuáles se queda corto (esto justifica narrativamente por qué hace falta EIF más adelante).
- Hacer lo mismo (visualización sano vs. fallo) también para la señal de vibración, aunque el análisis de armónicos en sí se aplique solo a la eléctrica.

## 2. Notebooks 02, 03 y 04 — Modelos (Isolation Forest, EIF, Transformers)

Para que la comparación entre modelos sea justa, **usar la misma extracción de características en los tres**, salvo que la investigación bibliográfica (ver punto 3) justifique lo contrario para vibración.

Cada uno de estos notebooks debe entrenar y evaluar **tres variantes de fuente de señal**:

- Solo señal eléctrica
- Solo señal de vibración
- Combinación (modelo híbrido eléctrica + vibración)

Y dejar explícito en el notebook una tabla/resumen con métricas de cada variante, para que luego el script de comparación (punto 5) pueda consumir esos resultados.

Estructura sugerida dentro de cada notebook:

- Explicación teórica breve del modelo (qué es, por qué se usa, en qué se diferencia del anterior — p.ej. EIF vs. IF tradicional, o Transformer vs. ambos).
- Extracción de features (ver punto 3).
- Entrenamiento sobre cada una de las 3 variantes de fuente de señal.
- Evaluación (métricas de detección de anomalías: precisión, recall, F1, AUC, etc., las que apliquen).
- Guardar resultados/métricas en un formato reutilizable (csv/json) para el script de comparación.
- Conclusiones parciales del notebook.

## 3. Extracción de características — función `espectro_db`

- Usar `espectro_db` como extractor de features principal.
- **Investigación bibliográfica obligatoria**: comprobar en la literatura si `espectro_db` (espectro en dB) es adecuado tanto para señal eléctrica como para señal de vibración, o si para vibración existe un método más apropiado (p. ej. envolvente, RMS por bandas, otros descriptores típicos de análisis vibracional). Documentar la conclusión con referencias antes de decidir qué extractor usar en cada tipo de señal.
- Si la literatura indica que conviene un extractor distinto para vibración, ajustar el pipeline de features de forma que cada tipo de señal use el extractor más adecuado, dejando constancia de la justificación en el notebook.

## 4. Consideraciones de diseño para el código

- Extraer a un módulo común (`utils/` o similar) las funciones compartidas entre notebooks: carga de datos, `espectro_db`, extracción de features, funciones de evaluación/métricas. Los notebooks importan de ahí en vez de duplicar código, para que la comparación sea consistente y el mantenimiento sea sencillo.
- Fijar semillas aleatorias para reproducibilidad.
- Guardar los resultados de cada notebook en una carpeta común de resultados (p. ej. `resultados/<modelo>_<fuente_senal>.json`).

## 5. Script/notebook de comparación final

- Cargar los resultados guardados por los 4 notebooks anteriores.
- Comparar: (a) baseline armónicos vs. IF vs. EIF vs. Transformers; (b) dentro de cada modelo, eléctrica vs. vibración vs. combinada.
- Generar tablas y gráficas comparativas (barras de métricas, etc.).
- Redactar conclusiones: qué combinación modelo+fuente de señal funciona mejor y por qué, apoyándose en lo observado en cada notebook individual.

## Notas finales

- Todo el trabajo es investigativo (TFM), así que prioriza la claridad explicativa sobre la brevedad del código: cada decisión debe estar justificada en el propio notebook.
- Si algún notebook requiere una versión previa exploratoria (visualización sano vs. fallo) que no esté ya cubierta en el notebook 01, inclúyela también en 02/03/04 antes de entrenar, para mantener el contexto visual en cada documento.
