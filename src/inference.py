#!/usr/bin/env python3
"""
Inference functionality for the Grateful Dead show dating model.
"""

import datetime
from typing import Dict, Union

import librosa
import torch
import torch.nn.functional as F

from models.dead_model import DeadShowDatingModel


def predict_date(
    model: DeadShowDatingModel,
    audio_path: str,
    base_date: datetime.date = datetime.date(1968, 1, 1),
    target_sr: int = 24000,
    device: Union[str, torch.device] = "cuda",
) -> Dict:
    """
    Run inference on a single audio file to predict its date.

    Args:
        model: Trained model
        audio_path: Path to audio file
        base_date: Base date for conversion
        target_sr: Target sample rate
        device: Device to run inference on

    Returns:
        Dictionary with prediction results
    """
    model.eval()

    try:
        # Load and preprocess audio
        y, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        y = y[: target_sr * 15]

        audio = torch.tensor(y, dtype=torch.float).unsqueeze(0)

        if audio.size(1) < target_sr * 15:
            audio = torch.nn.functional.pad(audio, (0, target_sr * 15 - audio.size(1)))

        audio = audio.to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(audio)

        # Process results
        pred_days = outputs["days"].item()
        pred_date = base_date + datetime.timedelta(days=int(pred_days))

        era_logits = outputs["era_logits"][0]
        pred_era = torch.argmax(era_logits).item()
        era_probs = F.softmax(era_logits, dim=0).cpu().numpy()

        era_names = {
            0: "Primal Dead (1965-1971)",
            1: "Europe 72 through Wall of Sound (1972-1974)",
            2: "Hiatus Return through Egypt (1975-1979)",
            3: "Brent Era (1980-1990)",
            4: "Bruce/Vince Era (1990-1995)",
        }

        return {
            "predicted_date": pred_date,
            "predicted_days_since_base": pred_days,
            "predicted_era": pred_era,
            "era_name": era_names[pred_era],
            "era_probabilities": {
                era_names[i]: prob for i, prob in enumerate(era_probs)
            },
        }

    except Exception as e:
        return {"error": str(e)}
