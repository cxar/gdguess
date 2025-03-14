#!/usr/bin/env python3
"""
Model architecture for the Grateful Dead show dating model.
"""

from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class ResidualBlock(nn.Module):
    """Residual block with two convolutions and a skip connection."""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout2d(0.1)  # Spatial dropout

        # Skip connection projection if needed
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.gelu(out)

        return out


class FrequencyTimeAttention(nn.Module):
    """
    Multi-head attention that explicitly models frequency and time relationships.
    Better suited for spectrogram data than standard attention.
    """

    def __init__(self, dim, heads=4, dropout=0.1):
        super(FrequencyTimeAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert (
            self.head_dim * heads == dim
        ), "Dimension must be divisible by number of heads"

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Relative position encoding
        self.pos_embedding = nn.Parameter(torch.randn(64, dim))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Create positional offset based on sequence length
        pos_offset = torch.arange(seq_len, device=x.device)
        pos_offset = pos_offset.unsqueeze(0) - pos_offset.unsqueeze(
            1
        )  # [seq_len, seq_len]
        pos_offset = pos_offset + 32  # Center the offset (assumed max length of 64)
        pos_offset = torch.clamp(pos_offset, 0, 63)  # Clamp to valid indices

        # Get relative position embeddings
        rel_pos = self.pos_embedding[pos_offset]  # [seq_len, seq_len, dim]

        # Get query, key, value
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(
                batch_size, seq_len, self.heads, self.head_dim
            ).transpose(1, 2),
            qkv,
        )

        # Calculate attention scores
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Add relative position bias
        rel_pos = rel_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)
        rel_pos = rel_pos.reshape(
            batch_size, seq_len, seq_len, self.heads, self.head_dim
        )
        rel_pos = rel_pos.permute(0, 3, 1, 2, 4)
        rel_pos_bias = torch.matmul(q.unsqueeze(3), rel_pos.transpose(-2, -1)).squeeze(
            3
        )
        attn = attn + rel_pos_bias

        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        out = self.to_out(out)

        return out


class DeadShowDatingModel(nn.Module):
    """
    Neural network model for dating Grateful Dead shows based on audio snippets.
    Predicts both the date (as days since a base date) and the era of the show.
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the model.

        Args:
            sample_rate: Audio sample rate to use for spectrograms
        """
        super(DeadShowDatingModel, self).__init__()

        # Audio preprocessing config
        self.sample_rate = sample_rate

        # Mel spectrogram transformation
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=8000,
        )

        # Convolutional feature extraction with residual connections
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),  # Replace ReLU with GELU
            nn.MaxPool2d(kernel_size=2),
            # Block 2 with residual
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualBlock(64, 64),  # Add residual block
            nn.MaxPool2d(kernel_size=2),
            # Block 3 with residual
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResidualBlock(128, 128),  # Add residual block
            nn.MaxPool2d(kernel_size=2),
            # Block 4 with residual
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            ResidualBlock(256, 256),  # Add residual block
            nn.MaxPool2d(kernel_size=2),
        )

        # Era detection module
        self.era_classifier = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5),  # 5 main Grateful Dead eras
        )

        # Self-attention for temporal patterns
        self.self_attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)

        # Add frequency-time attention
        self.freq_time_attention = FrequencyTimeAttention(256, heads=4)

        # Replace GRU with LSTM for improved temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Final regression for date prediction
        self.regressor = nn.Sequential(
            nn.Linear(256 * 2 + 5, 256),  # LSTM output + era features
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),  # Predicts number of days since base date
        )

    def forward(self, x: Union[torch.Tensor, Dict]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the model, handles both raw audio and batch dictionary inputs.

        Args:
            x: Input data, either a raw audio tensor or a dictionary with relevant keys

        Returns:
            Dictionary containing 'days' (date prediction) and 'era_logits' (era classification logits)
        """
        try:
            # Get the device of the model
            model_device = next(self.parameters()).device
            batch_size = 0  # Will be set based on input

            # Handle both tensor inputs and dictionary batch inputs
            if isinstance(x, dict):
                # Check if we're using precomputed mel spectrograms
                if "mel_spec" in x:
                    # Use the precomputed mel spectrogram directly
                    mel_spec = x["mel_spec"].to(model_device, non_blocking=True)
                    batch_size = mel_spec.shape[0]

                    # Extract features directly from mel spec
                    features = self.feature_extractor(mel_spec)
                # Otherwise use audio
                elif "audio" in x:
                    audio = x["audio"].to(model_device, non_blocking=True)
                    batch_size = audio.shape[0]

                    # Convert to mel spectrogram
                    with torch.no_grad():
                        audio_mel = self.mel_spec(audio)
                        audio_mel = torch.log(audio_mel + 1e-4)
                        audio_mel = audio_mel.unsqueeze(1)  # Add channel dimension

                    features = self.feature_extractor(audio_mel)
                elif "raw_audio" in x:
                    audio = x["raw_audio"].to(model_device, non_blocking=True)
                    batch_size = audio.shape[0]

                    # Convert to mel spectrogram
                    with torch.no_grad():
                        audio_mel = self.mel_spec(audio)
                        audio_mel = torch.log(audio_mel + 1e-4)
                        audio_mel = audio_mel.unsqueeze(1)  # Add channel dimension

                    features = self.feature_extractor(audio_mel)
                else:
                    # Fallback case for older code paths
                    batch_size = len(x["label"])
                    dummy_days = torch.zeros((batch_size,), device=model_device)
                    dummy_logits = torch.zeros((batch_size, 5), device=model_device)
                    return {"days": dummy_days, "era_logits": dummy_logits}
            else:
                # Direct tensor input (e.g., from inference) - assume raw audio
                audio = x.to(model_device, non_blocking=True)
                batch_size = audio.shape[0]

                # Convert to mel spectrogram
                with torch.no_grad():
                    audio_mel = self.mel_spec(audio)
                    audio_mel = torch.log(audio_mel + 1e-4)
                    audio_mel = audio_mel.unsqueeze(1)  # Add channel dimension

                features = self.feature_extractor(audio_mel)

            # Classify era
            era_logits = self.era_classifier(features)
            era_features = F.softmax(era_logits, dim=1)

            if torch.isnan(era_features).any():
                era_features = torch.where(
                    torch.isnan(era_features),
                    torch.zeros_like(era_features),
                    era_features,
                )

            # Reshape for temporal processing
            b, c, h, w = features.shape
            features_temporal = features.permute(0, 2, 3, 1).reshape(b, h * w, c)

            # Apply self-attention
            attn_output, _ = self.self_attention(
                features_temporal,
                features_temporal,
                features_temporal,
                need_weights=False,
            )

            # Apply frequency-time attention for spectral patterns
            attn_output = self.freq_time_attention(attn_output)

            # Process with LSTM instead of GRU
            lstm_out, (hidden, _) = self.lstm(attn_output)

            # Get final hidden state
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)

            # Combine with era features for regression
            combined = torch.cat([hidden_concat, era_features], dim=1)
            days_prediction = self.regressor(combined)

            if torch.isnan(days_prediction).any():
                days_prediction = torch.where(
                    torch.isnan(days_prediction),
                    torch.zeros_like(days_prediction),
                    days_prediction,
                )

            return {"days": days_prediction.squeeze(-1), "era_logits": era_logits}
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            # Use the model's device for dummy tensors
            model_device = next(self.parameters()).device
            dummy_days = torch.zeros((batch_size,), device=model_device)
            dummy_logits = torch.zeros((batch_size, 5), device=model_device)
            return {"days": dummy_days, "era_logits": dummy_logits}
