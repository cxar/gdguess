#!/usr/bin/env python3
"""
Audio augmentation techniques specific to Grateful Dead shows.
"""

import numpy as np


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

        # Tape wear simulation
        if np.random.random() > 0.6:

            def tape_wear(x):
                window_size = int(self.sample_rate * 0.002)
                window = np.ones(window_size) / window_size
                filtered_x = np.convolve(x, window, mode="same")
                return filtered_x

            augmentations.append(tape_wear)

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
