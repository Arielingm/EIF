"""
Extracción de características espectrales por ventana de señal.

Dos extractores, uno por tipo de señal (ver justificación bibliográfica
en 01_baseline_armonicos.ipynb / README de resultados de la tarea 3):

- `espectro_db`            → señales eléctricas (u, v, w).
  MCSA (Motor Current Signature Analysis) clásico: el fallo se manifiesta
  como armónicos/bandas laterales alrededor de la frecuencia de red en el
  espectro de amplitud de la corriente. El espectro FFT directo ya expone
  esas componentes sin necesidad de demodulación.

- `envolvente_espectro_db` → señales de vibración (front/rear/housing).
  Los defectos de rodamiento (BPFO/BPFI/BSF/FTF) generan impactos de
  amplitud modulada sobre una portadora de alta frecuencia (resonancia
  estructural), que quedan enmascarados en el espectro FFT directo. La
  técnica de referencia en la literatura es el envelope spectrum: se extrae
  la envolvente de amplitud vía transformada de Hilbert y se calcula su FFT,
  lo que revierte la modulación y deja las frecuencias de fallo como picos
  claros de baja frecuencia.
"""

import numpy as np
from scipy.signal import hilbert


def espectro_db(segmento: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Espectro de amplitud normalizado en dB:

        20 * log10( |FFT(x * Hanning)| / max(|FFT(x * Hanning)|) )

    Usado para señales eléctricas (u, v, w).
    """
    ventana_hanning = np.hanning(len(segmento))
    espectro        = np.abs(np.fft.rfft(segmento * ventana_hanning))
    maximo          = np.max(espectro)

    if maximo < 1e-12:
        return np.full(n_bins, -120.0)

    espectro_norm = espectro / maximo
    espectro_db   = 20.0 * np.log10(espectro_norm + 1e-12)

    return espectro_db[:n_bins]


def envolvente_espectro_db(segmento: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Envelope spectrum normalizado en dB:

        envolvente = |hilbert(x)|                    (demodulación de amplitud)
        20 * log10( |FFT((envolvente - media) * Hanning)| / max(...) )

    Se resta la media de la envolvente antes de la FFT para eliminar el pico
    DC dominante y dejar visibles las frecuencias de modulación (fallo).
    Usado para señales de vibración.
    """
    envolvente = np.abs(hilbert(segmento))
    envolvente = envolvente - envolvente.mean()

    ventana_hanning = np.hanning(len(envolvente))
    espectro        = np.abs(np.fft.rfft(envolvente * ventana_hanning))
    maximo          = np.max(espectro)

    if maximo < 1e-12:
        return np.full(n_bins, -120.0)

    espectro_norm = espectro / maximo
    espectro_db   = 20.0 * np.log10(espectro_norm + 1e-12)

    return espectro_db[:n_bins]
