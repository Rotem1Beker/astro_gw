# FFT, חלונות זמן, נרמול
import numpy as np

#פונקציה שמקבלת אות בזמן ומחזירה ייצוג תדרי מנורמל – מוכן להזנה לרשת.כת
def preprocess(x):
    X = np.abs(np.fft.rfft(x))
    X = (X - X.mean()) / (X.std() + 1e-6)
    return X.astype(np.float32)

def whiten(strain):
    return (strain - np.mean(strain)) / np.std(strain)

"""
Preprocessing of gravitational-wave time series data.

This module performs basic signal preprocessing prior to feeding the data
into a neural network. The preprocessing includes:
1. Transformation from the time domain to the frequency domain using a
   real-valued FFT (rFFT).
2. Conversion to an amplitude spectrum by taking the absolute value,
   discarding phase information.
3. Standardization (zero mean, unit variance) to stabilize neural network
   training.

This is a simplified preprocessing pipeline intended for simulation-based
inference experiments. It does not include windowing, whitening, or detector
noise PSD modeling, which are typically required for realistic gravitational-
wave data analysis.
"""
