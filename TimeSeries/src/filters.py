import numpy as np

def moving_average(signal, window_size=5):
    padded = np.pad(signal, (window_size//2, window_size - 1 - window_size//2), mode='edge')
    weights = np.ones(window_size) / window_size
    return np.convolve(padded, weights, mode='valid')

def median_filter(signal, window_size=5):
    pad_size = window_size // 2
    padded = np.pad(signal, (pad_size, pad_size), mode='edge')
    result = np.zeros_like(signal)
    for i in range(len(signal)):
        result[i] = np.median(padded[i:i+window_size])
    return result

def classical_decomposition(signal, period):

    window = period if period % 2 != 0 else period + 1
    trend = moving_average(signal, window_size=window)
    
    detrended = signal - trend
    
    seasonal = np.zeros_like(signal)
    season_averages = np.zeros(period)
    counts = np.zeros(period)
    
    for i in range(len(signal)):
        if not np.isnan(detrended[i]):
            season_averages[i % period] += detrended[i]
            counts[i % period] += 1
            
    
    counts[counts == 0] = 1
    season_averages /= counts
    season_averages -= np.mean(season_averages)
    
    for i in range(len(signal)):
        seasonal[i] = season_averages[i % period]
        
    residual = signal - trend - seasonal
    return trend, seasonal, residual

def kalman_filter_1d(signal, process_variance=1e-3, measurement_variance=1.0):

    n = len(signal)
    x_hat = np.zeros(n)
    P = np.zeros(n)
    
    
    valid_idx = np.where(~np.isnan(signal))[0]
    if len(valid_idx) > 0:
        x_hat[0] = signal[valid_idx[0]]
    else:
        x_hat[0] = 0
    P[0] = 1.0
    
    for k in range(1, n):
        x_pred = x_hat[k-1]
        P_pred = P[k-1] + process_variance
        
        if np.isnan(signal[k]):
            
            x_hat[k] = x_pred
            P[k] = P_pred
        else:
            
            K = P_pred / (P_pred + measurement_variance)
            x_hat[k] = x_pred + K * (signal[k] - x_pred)
            P[k] = (1 - K) * P_pred
            
    return x_hat

def fourier_low_pass_filter(signal, sample_rate, cutoff_freq):
    
    clean_sig = linear_interpolate_missing(signal)
    
    freqs = np.fft.fftfreq(len(clean_sig), d=1/sample_rate)
    fft_vals = np.fft.fft(clean_sig)
    
    fft_vals_filtered = fft_vals.copy()
    fft_vals_filtered[np.abs(freqs) > cutoff_freq] = 0
    
    filtered_signal = np.fft.ifft(fft_vals_filtered)
    return np.real(filtered_signal)

def linear_interpolate_missing(signal):
    sig = signal.copy()
    nans = np.isnan(sig)
    
    nan_indices = np.nonzero(nans)[0]
    valid_indices = np.nonzero(~nans)[0]
    
    if len(valid_indices) == 0:
        return np.zeros_like(sig)
        
    sig[nans] = np.interp(nan_indices, valid_indices, sig[valid_indices])
    return sig
