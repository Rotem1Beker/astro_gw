# preprocess.py
# Fixed preprocessing that preserves information

import numpy as np

def preprocess(strain):
    """
    Preprocessing של אות LIGO
    
    Args:
        strain: time-domain signal
    
    Returns:
        processed: frequency-domain features (1025,)
    """
    # FFT
    fft = np.fft.rfft(strain)
    fft_abs = np.abs(fft)
    
    # 🔧 CRITICAL FIX: Use log-scale with FIXED reference
    # This preserves relative differences between signals!
    fft_log = np.log10(fft_abs + 1e-25)  # log scale with fixed floor
    
    # Clip to reasonable range (avoid extreme outliers)
    fft_clipped = np.clip(fft_log, -25, -10)
    
    # Scale to [0, 1] range using FIXED bounds
    fft_normalized = (fft_clipped + 25) / 15.0  # maps [-25, -10] -> [0, 1]
    
    return fft_normalized