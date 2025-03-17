"""
Preprocessing module for the Grateful Dead show dating model.
Contains tools for audio feature extraction and preprocessing.
"""

from .turboprocess import (
    find_audio_files,
    init_gpu, 
    init_mps,
    optimize_for_device,
    get_feature_extractor,
    gpu_hpss,
    gpu_spectral_contrast,
    gpu_chroma, 
    gpu_onset_strength,
    create_chroma_filter_bank,
    process_batch,
    process_multiple_files
)

from .preprocess import preprocess_dataset

__all__ = [
    'find_audio_files',
    'init_gpu',
    'init_mps',
    'optimize_for_device',
    'get_feature_extractor',
    'gpu_hpss',
    'gpu_spectral_contrast',
    'gpu_chroma',
    'gpu_onset_strength',
    'create_chroma_filter_bank',
    'process_batch',
    'process_multiple_files',
    'preprocess_dataset'
] 