"""
Neural network model definitions for the Grateful Dead show dating project.
"""

# Main model
from .dead_model import DeadShowDatingModel

# Components
from .components import (
    AudioFeatureExtractor,
    FrequencyTimeAttention,
    ResidualBlock,
    PositionalEncoding
)

# Feature extractors
from .feature_extractors import (
    ParallelFeatureNetwork,
    SeasonalPatternModule
)

# Loss functions
from .losses import (
    UncertaintyLoss,
    PeriodicityLoss,
    DynamicTaskLoss,
    CombinedDeadLoss
)

__all__ = [
    # Main model
    "DeadShowDatingModel",
    
    # Components
    "AudioFeatureExtractor",
    "FrequencyTimeAttention",
    "ResidualBlock",
    "PositionalEncoding",
    
    # Feature extractors
    "ParallelFeatureNetwork",
    "SeasonalPatternModule",
    
    # Loss functions
    "UncertaintyLoss",
    "PeriodicityLoss", 
    "DynamicTaskLoss",
    "CombinedDeadLoss"
]
