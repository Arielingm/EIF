# 🌀 Procesamiento de Señales y Extracción de Características

Este proyecto permite procesar archivos `.mat` que contienen señales multicanal (como las provenientes de sensores) para extraer características estadísticas y no lineales. Los datos son posteriormente normalizados mediante Z-score para que puedan ser utilizados en modelos de machine learning para tareas como detección de fallos.

---

## 📁 Estructura del Proyecto

```plaintext
.
├── AllData/
│   ├── DatosSanos_Entrenamiento/   # Archivos .mat del set de entrenamiento
│   ├── Fallo1/                     # Archivos .mat con señales con fallo
│   ├── ProcesadoMatrizZ/          # Directorio de salida para CSVs procesados
│   └── CSV/                       # Medias y desviaciones estándar
├── extraer_features_senales.py    # Script principal de procesamiento
├── codigo_entreno.ipynb           # Notebook con pruebas y visualizaciones
├── requirements.txt               # Dependencias del proyecto
├── .gitignore                     # Archivos y carpetas ignoradas por Git
└── README.md                      # Este archivo

```
## ⚙️ Funcionalidades

El script principal (`extraer_features_senales.py`) realiza:

- Carga de archivos `.mat`
- División temporal de las señales en ventanas
- Extracción de características por canal:
  - Dimensión fractal de Katz
  - Entropía de permutación
  - Curtosis
- Etiquetado de muestras (normal/fallo)
- Normalización con Z-score
- Exportación a `.csv`

---

## 🚀 Cómo usar

### 1. Crear medias y desviaciones (modo entrenamiento)

```python
procesar_datos(
    ruta_entrada='./AllData/DatosSanos_Entrenamiento',
    ruta_salida_csv='./AllData/ProcesadoMatrizZ/DatosSanos75.csv',
    modo='entrenamiento',
    path_medias='./AllData/CSV/MediaDatosSanos75.csv'
)


``` 

### 2. Procesar otros sets con normalización (modo evaluación)

```python
procesar_datos(
    ruta_entrada='./AllData/Fallo1',
    ruta_salida_csv='./AllData/ProcesadoMatrizZ/Fallo1.csv',
    modo='evaluacion',
    path_medias='./AllData/CSV/MediaDatosSanos75.csv'
)
```
También puedes repetir este proceso para otros conjuntos de datos, por ejemplo:

- `./AllData/Validacion`
- `./AllData/TesteoSanos`
- `./AllData/Fallo2`
- `./AllData/Fallo3`
- `./AllData/Fallo4`
- etc.

## 🧪 Requisitos

Asegúrate de tener Python instalado (preferiblemente 3.8 o superior).

Instala las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

## 📌 Notas

- Puedes adaptar el tamaño de las ventanas modificando el parámetro `f * 1.875`, que representa **aproximadamente medio giro de la pala**.  
  Si el tiempo de rotación cambia, ajusta este valor para mantener la coherencia temporal en la extracción de características.

## 👨‍💻 Autores
Ariel Gonzalez
Abel Gomez

- **Iván Ariel González Moreira**  
  📧 Contacto: *ivargonzm@gmail.com*

- **Abel Gómez**  
  📧 Contacto: *abel.gomez@example.com*
