import numpy as np
import pandas as pd

def calculate_rmse(clean, estimated):
                                                          
    mask = ~np.isnan(clean) & ~np.isnan(estimated)
    if not np.any(mask):
        return np.inf
    return np.sqrt(np.mean((clean[mask] - estimated[mask])**2))

def calculate_mae(clean, estimated):
                                                       
    mask = ~np.isnan(clean) & ~np.isnan(estimated)
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs(clean[mask] - estimated[mask]))

def calculate_snr(clean, estimated):
                                                                            
    mask = ~np.isnan(clean) & ~np.isnan(estimated)
    if not np.any(mask):
        return -np.inf
    signal_power = np.mean(clean[mask]**2)
    noise_power = np.mean((clean[mask] - estimated[mask])**2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)
