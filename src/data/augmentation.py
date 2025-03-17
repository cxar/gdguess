#!/usr/bin/env python3
"""
Audio augmentation techniques specific to Grateful Dead shows.
"""

import numpy as np
import scipy.signal
import librosa


class DeadShowAugmenter:
    """
    Applies Grateful Dead-specific audio augmentations to simulate different
    eras, recording conditions, and tape degradation.
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the augmenter.

        Args:
            sample_rate: Audio sample rate, used for time-based effects
        """
        self.sample_rate = sample_rate

    def augment(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Grateful Dead specific augmentations.

        Args:
            audio: Input audio as a numpy array

        Returns:
            Augmented audio as a numpy array
        """
        augmentations = []

        # VERY SMALL PITCH SHIFTING
        if np.random.random() > 0.75:
            def pitch_shift(x):
                # Random pitch shift between -0.5 and 0.5 semitones
                n_steps = np.random.uniform(-0.5, 0.5)
                return librosa.effects.pitch_shift(x, sr=self.sample_rate, n_steps=n_steps)
            
            augmentations.append(pitch_shift)

        # VERY SMALL TIME STRETCHING
        if np.random.random() > 0.8:
            def time_stretch(x):
                # Random time stretch factor between 0.95 and 1.05
                rate = np.random.uniform(0.95, 1.05)
                # Use phase vocoder for better quality
                return librosa.effects.time_stretch(x, rate=rate)
            
            augmentations.append(time_stretch)

        # Audience simulation
        if np.random.random() > 0.7:
            def audience_sim(x):
                impulse = np.exp(-np.linspace(0, 10, int(0.3 * self.sample_rate)))
                reverb = np.convolve(x, impulse, mode="full")[: len(x)]
                noise_level = np.random.uniform(0.01, 0.05)
                noise = np.random.randn(len(x)) * noise_level
                mix_ratio = np.random.uniform(0.5, 0.8)
                return x * mix_ratio + reverb * (1 - mix_ratio) + noise

            augmentations.append(audience_sim)

        # Enhanced tape degradation simulation
        if np.random.random() > 0.6:
            def enhanced_tape_wear(x):
                # Combine multiple degradation effects
                
                # 1. Basic tape wear - low pass filter
                window_size = int(self.sample_rate * 0.002)
                window = np.ones(window_size) / window_size
                filtered_x = np.convolve(x, window, mode="same")
                
                # 2. Add wow and flutter - modulate the signal with LFO
                if np.random.random() > 0.5:
                    # Wow effect - slow frequency modulation (0.5-2 Hz)
                    wow_rate = np.random.uniform(0.5, 2.0)
                    wow_depth = np.random.uniform(0.0005, 0.002)
                    wow_lfo = np.sin(2 * np.pi * wow_rate * np.arange(len(x)) / self.sample_rate)
                    
                    # Time domain modulation
                    indices = np.arange(len(x)) + wow_depth * self.sample_rate * wow_lfo
                    indices = np.clip(indices, 0, len(x) - 1).astype(np.int32)
                    filtered_x = filtered_x[indices]
                
                # 3. Add dropouts - simulate tape imperfections
                if np.random.random() > 0.7:
                    dropout_count = np.random.randint(1, 5)
                    for _ in range(dropout_count):
                        # Random position and duration for dropout
                        pos = np.random.randint(0, len(x) - 1)
                        duration = np.random.randint(int(0.01 * self.sample_rate), 
                                                    int(0.05 * self.sample_rate))
                        end_pos = min(pos + duration, len(x))
                        
                        # Apply dropout (reduce amplitude)
                        dropout_factor = np.random.uniform(0.1, 0.5)
                        filtered_x[pos:end_pos] *= dropout_factor
                
                # 4. Add tape hiss - shaped noise profile
                if np.random.random() > 0.5:
                    noise_level = np.random.uniform(0.001, 0.01)
                    noise = np.random.randn(len(x)) * noise_level
                    
                    # Shape noise to have more high frequency content (tape hiss)
                    b, a = scipy.signal.butter(1, 0.1, 'highpass', analog=False)
                    shaped_noise = scipy.signal.filtfilt(b, a, noise)
                    
                    filtered_x += shaped_noise
                
                # 5. Add subtle saturation
                if np.random.random() > 0.5:
                    saturation = np.random.uniform(1.0, 2.0)
                    filtered_x = np.tanh(filtered_x * saturation) / np.tanh(saturation)
                
                return filtered_x

            augmentations.append(enhanced_tape_wear)

        # Era-specific EQ
        if np.random.random() > 0.5:
            era = np.random.choice(["early", "seventies", "eighties", "nineties"])

            def era_eq(x):
                if era == "early":
                    X = np.fft.rfft(x)
                    freq = np.fft.rfftfreq(len(x), 1 / self.sample_rate)
                    mask = np.logical_and(freq >= 50, freq <= 7000)
                    X_filtered = X * mask
                    x_filtered = np.fft.irfft(X_filtered, len(x))
                    x_filtered = np.tanh(x_filtered * 1.2)
                    return x_filtered
                elif era == "seventies":
                    X = np.fft.rfft(x)
                    freq = np.fft.rfftfreq(len(x), 1 / self.sample_rate)
                    mask = np.ones_like(X, dtype=float)
                    midrange_mask = np.logical_and(freq >= 300, freq <= 2500)
                    mask[midrange_mask] = 1.3
                    X_filtered = X * mask
                    x_filtered = np.fft.irfft(X_filtered, len(x))
                    x_filtered = (
                        np.sign(x_filtered)
                        * np.log(1 + 5 * np.abs(x_filtered))
                        / np.log(6)
                    )
                    return x_filtered
                elif era == "eighties":
                    X = np.fft.rfft(x)
                    freq = np.fft.rfftfreq(len(x), 1 / self.sample_rate)
                    mask = np.ones_like(X, dtype=float)
                    high_mask = freq >= 5000
                    mask[high_mask] = 1.2
                    X_filtered = X * mask
                    x_filtered = np.fft.irfft(X_filtered, len(x))
                    bits = np.random.randint(10, 16)
                    x_filtered = np.round(x_filtered * (2**bits)) / (2**bits)
                    return x_filtered
                elif era == "nineties":
                    X = np.fft.rfft(x)
                    freq = np.fft.rfftfreq(len(x), 1 / self.sample_rate)
                    mask = np.ones_like(X, dtype=float)
                    mask[freq < 100] = 1.1
                    mask[freq > 8000] = 1.1
                    X_filtered = X * mask
                    x_filtered = np.fft.irfft(X_filtered, len(x))
                    return x_filtered
                return x

            augmentations.append(era_eq)

        # Apply all selected augmentations
        augmented_audio = audio.copy()
        for aug_func in augmentations:
            augmented_audio = aug_func(augmented_audio)

        # Normalize the output
        if np.max(np.abs(augmented_audio)) > 0:
            augmented_audio = augmented_audio / np.max(np.abs(augmented_audio))

        return augmented_audio
