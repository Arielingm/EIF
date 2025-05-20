import csv
import numpy as np
import scipy.io as sio
import pandas as pd
import os
from itertools import permutations
from math import factorial
from math import log2
from scipy.stats import entropy
from scipy.stats import kurtosis


# --- Funciones auxiliares ---
"""
 ####
 Funcion Para DIM FRACTAL
 ####
"""

def calcular_dim_fractal_katz(signal):
    # Calculate the total length of the signal
    L = np.sum(np.sqrt(1 + np.square(np.diff(signal))))

    # Calculate the average distance between points
    N = len(signal)-1
    a = L / N #(len(signal) - 1)

    # Calculate the maximum distance between points
    puntox = np.array(list(range(1, len(signal)+1)))
    distances = np.sqrt(np.square(puntox - puntox[0])+ np.square(signal - signal[0]))
    d = np.max(distances)

    # Calculate the Katz Fractal Dimension
    return (np.log10(N) / (np.log10(d/L) + np.log10(N)))

"""
 ####
 Funcion Para Entropia de Permitacion
 ####
"""

def permutacionEntropyPRO(signal, m=3,normalizar=True):
    N = len(signal) - m + 1
    P = np.zeros(factorial(m))
    perms = np.array(list(permutations(range(m))))
    for i in range(N):
        idx = np.argsort(signal[i:i+m])
        perm_idx = np.where((perms == idx).all(axis=1))[0][0]
        P[perm_idx] += 1
    entropia = entropy(P, base=2)
    if(normalizar):
        return entropia/log2(factorial(m))
    return entropia

"""
 ####
 Funcion mean, standard_deviation
 ####
"""

def Media_Desviacion(values):
    # Calculate the Standard Deviation in Python
    mean = sum(values) / len(values)
    differences = [(value - mean)**2 for value in values]
    sum_of_differences = sum(differences)
    standard_deviation = (sum_of_differences / (len(values) - 1)) ** 0.5

    return mean,standard_deviation

"""
 ####
 Funcion Z-score https://datagy.io/python-z-score/
 ####
"""
def Zscores(values,mean, standard_deviation):
    # Calculate the z-score from scratch
    zscores = [(value - mean) / standard_deviation for value in values]
    return zscores

"""
"""
def cargar_medias_desviaciones(path_csv):
    with open(path_csv, 'r') as f:
        return [(float(fila[0]), float(fila[1])) for fila in csv.reader(f)]
    
"""
"""
def guardar_medias_desviaciones(medias_desv, path_csv):
    with open(path_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(medias_desv)


# --- Función principal de procesamiento ---

def procesar_datos(ruta_entrada, ruta_salida_csv, modo="entrenamiento", path_medias=None):
    """
    modo: "entrenamiento" -> calcula medias y desv y las guarda.
          "evaluacion" -> usa medias y desv precalculadas de path_medias
    """
    archivos = [f for f in os.listdir(ruta_entrada) if f.endswith('.mat')]
    df_total = pd.DataFrame({})
    mean_ds = []

    for ind, archivo in enumerate(archivos):
        a, b = archivo.split("_")
        data = sio.loadmat(os.path.join(ruta_entrada, archivo))
        df = pd.DataFrame(data["data"])
        f = len(df.index) / 60

        for j in range(0, len(df.index), int(f * 1.875)):
            subdf = df.iloc[j:j + int(f * 1.875)]
            if subdf.empty: continue

            results = np.empty((subdf.shape[1], 3))
            for i in range(subdf.shape[1]):
                signal = subdf.iloc[:, i].values
                results[i, 0] = calcular_dim_fractal_katz(signal)
                results[i, 1] = permutacionEntropyPRO(signal, 6, False)
                results[i, 2] = kurtosis(signal)

            fila = results.reshape(1, -1)
            etiqueta = 1 if a == "Data" else 0
            fila = np.insert(fila, fila.shape[1], etiqueta, axis=1)
            df_total = pd.concat([df_total, pd.DataFrame(fila)], ignore_index=True)

        print(f"Procesado {ind+1}/{len(archivos)}")

    # Normalización por z-score
    if modo == "entrenamiento":
        for col in range(df_total.shape[1]-1):
            values = df_total[col]
            mean, std = Media_Desviacion(values)
            mean_ds.append((mean, std))
            df_total[col] = Zscores(values, mean, std)
        guardar_medias_desviaciones(mean_ds, path_medias)
    else:
        medias = cargar_medias_desviaciones(path_medias)
        for col in range(df_total.shape[1]-1):
            mean, std = medias[col]
            df_total[col] = Zscores(df_total[col], mean, std)

    # Guardar el CSV procesado
    df_total.to_csv(ruta_salida_csv, index=False, header=False)


"""
EJEMPLO PARA ENTRENAMIENTO
"""

procesar_datos(
    ruta_entrada='./AllData/DatosSanos_Entrenamiento',
    ruta_salida_csv='./AllData/ProcesadoMatrizZ/DatosSanos75.csv',
    modo='entrenamiento',
    path_medias='./AllData/CSV/MediaDatosSanos75.csv'
)


"""
EJEMPLO PARA EVALUACIÓN (validación, test o fallos)
"""

procesar_datos(
    ruta_entrada='./AllData/Fallo1',
    ruta_salida_csv='./AllData/ProcesadoMatrizZ/Fallo1.csv',
    modo='evaluacion',
    path_medias='./AllData/CSV/MediaDatosSanos75.csv'
)
