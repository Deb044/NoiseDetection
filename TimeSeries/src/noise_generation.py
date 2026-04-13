import numpy as np

def generate_clean_signal(n_samples=1000, t_start=0, t_end=10, components=None):
    """
    Generate a clean time series signal from multiple sine wave components.
    
    Args:
        n_samples (int): Number of samples to generate.
        t_start (float): Start time.
        t_end (float): End time.
        components (list of dicts): List of dictionaries specifying components.
            Each dictionary should have 'freq' (Hz) and 'amplitude'.
            
    Returns:
        t (np.ndarray): Time vector.
        signal (np.ndarray): Generated clean signal.
    """
    if components is None:
        components = [
            {'freq': 1.0, 'amplitude': 1.0}, # 1 Hz wave
            {'freq': 0.2, 'amplitude': 2.0}, # Low freq background
            {'freq': 5.0, 'amplitude': 0.5}  # Higher freq detail
        ]
        
    t = np.linspace(t_start, t_end, n_samples)
    signal = np.zeros_like(t)
    
    for comp in components:
        signal += comp['amplitude'] * np.sin(2 * np.pi * comp['freq'] * t)
        
    return t, signal

def add_gaussian_noise(signal, mean=0, std=1.0):
    """Add Gaussian noise to the signal."""
    noise = np.random.normal(mean, std, signal.shape)
    return signal + noise

def add_spike_noise(signal, num_spikes=10, spike_amplitude=5.0):
    """Add random heavy spikes to the signal."""
    noisy_signal = signal.copy()
    indices = np.random.choice(len(signal), size=num_spikes, replace=False)
    
    # Randomly make spikes positive or negative
    signs = np.random.choice([-1, 1], size=num_spikes)
    noisy_signal[indices] += signs * spike_amplitude
    return noisy_signal

def add_drift_noise(signal, drift_type='linear', drift_strength=0.5):
    """Add a drift pattern (either linear or random walk)."""
    noisy_signal = signal.copy()
    n = len(signal)
    if drift_type == 'linear':
        # Add a linearly increasing/decreasing trend
        drift = np.linspace(0, drift_strength * n, n)
        noisy_signal += drift
    elif drift_type == 'random_walk':
        # Add a cumulative sum of random variables representing sensor drift
        drift = np.cumsum(np.random.normal(0, drift_strength, n))
        noisy_signal += drift
    return noisy_signal

def add_seasonal_noise(t, signal, freq=0.5, amplitude=2.0):
    """Add a separate seasonal pattern to the signal (e.g. daily temperature cycle superimposed)."""
    seasonal_artifact = amplitude * np.sin(2 * np.pi * freq * t)
    return signal + seasonal_artifact

def add_missing_data(signal, missing_pct=0.1, block_size=1):
    """
    Introduce missing data (NaN) into the signal.
    
    Args:
        missing_pct (float): Percentage of data to drop.
        block_size (int): Size of the contiguous blocks of missing data.
    """
    noisy_signal = signal.copy().astype(float)
    n = len(signal)
    num_missing = int(n * missing_pct)
    
    if block_size == 1:
        indices = np.random.choice(n, size=num_missing, replace=False)
        noisy_signal[indices] = np.nan
    else:
        num_blocks = max(1, num_missing // block_size)
        # Choose random start indices for blocks
        start_indices = np.random.choice(n - block_size, size=num_blocks, replace=False)
        for i in start_indices:
            noisy_signal[i:i+block_size] = np.nan
            
    return noisy_signal
