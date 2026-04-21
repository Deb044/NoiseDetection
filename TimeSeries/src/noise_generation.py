import numpy as np

def generate_clean_signal(n_samples=1000, t_start=0, t_end=10, components=None):
    if components is None:
        components = [
            {'freq': 1.0, 'amplitude': 1.0}, 
            {'freq': 0.2, 'amplitude': 2.0}, 
            {'freq': 5.0, 'amplitude': 0.5}  
        ]
        
    t = np.linspace(t_start, t_end, n_samples)
    signal = np.zeros_like(t)
    
    for comp in components:
        signal += comp['amplitude'] * np.sin(2 * np.pi * comp['freq'] * t)
        
    return t, signal

def add_gaussian_noise(signal, mean=0, std=1.0):
    noise = np.random.normal(mean, std, signal.shape)
    return signal + noise

def add_spike_noise(signal, num_spikes=10, spike_amplitude=5.0):
    noisy_signal = signal.copy()
    indices = np.random.choice(len(signal), size=num_spikes, replace=False)
    
    
    signs = np.random.choice([-1, 1], size=num_spikes)
    noisy_signal[indices] += signs * spike_amplitude
    return noisy_signal

def add_drift_noise(signal, drift_type='linear', drift_strength=0.5):
    noisy_signal = signal.copy()
    n = len(signal)
    if drift_type == 'linear':
        
        drift = np.linspace(0, drift_strength * n, n)
        noisy_signal += drift
    elif drift_type == 'random_walk':
        
        drift = np.cumsum(np.random.normal(0, drift_strength, n))
        noisy_signal += drift
    return noisy_signal

def add_seasonal_noise(t, signal, freq=0.5, amplitude=2.0):
    seasonal_artifact = amplitude * np.sin(2 * np.pi * freq * t)
    return signal + seasonal_artifact

def add_missing_data(signal, missing_pct=0.1, block_size=1):
    noisy_signal = signal.copy().astype(float)
    n = len(signal)
    num_missing = int(n * missing_pct)
    
    if block_size == 1:
        indices = np.random.choice(n, size=num_missing, replace=False)
        noisy_signal[indices] = np.nan
    else:
        num_blocks = max(1, num_missing // block_size)
        
        start_indices = np.random.choice(n - block_size, size=num_blocks, replace=False)
        for i in start_indices:
            noisy_signal[i:i+block_size] = np.nan
            
    return noisy_signal
