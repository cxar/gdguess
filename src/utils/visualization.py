#!/usr/bin/env python3
"""
Visualization and logging functions for the Grateful Dead show dating model.
"""

import datetime
import io
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard.writer import SummaryWriter


def log_prediction_samples(
    writer: SummaryWriter,
    true_days: List[float],
    pred_days: List[float],
    base_date: datetime.date,
    epoch: int,
    prefix: str = "train",
) -> None:
    """
    Log sample predictions to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        true_days: List of true day values
        pred_days: List of predicted day values
        base_date: Base date for conversion
        epoch: Current epoch number
        prefix: Prefix for TensorBoard tags
    """
    try:
        samples = min(10, len(true_days))
        indices = np.random.choice(len(true_days), samples, replace=False)
        headers = ["True Date", "Predicted Date", "Error (days)"]
        data = []

        for i in indices:
            try:
                true_date = base_date + datetime.timedelta(days=int(true_days[i]))
                pred_date = base_date + datetime.timedelta(days=int(pred_days[i]))
                error = abs(int(true_days[i]) - int(pred_days[i]))
                data.append(
                    [
                        true_date.strftime("%Y-%m-%d"),
                        pred_date.strftime("%Y-%m-%d"),
                        str(error),
                    ]
                )
            except (ValueError, OverflowError, TypeError) as e:
                print(f"Error processing prediction sample {i}: {e}")
                continue

        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in data:
            table += "| " + " | ".join(row) + " |\n"

        writer.add_text(f"{prefix}/date_predictions", table, epoch)
    except Exception as e:
        print(f"Error in log_prediction_samples: {e}")


def log_era_confusion_matrix(
    writer: SummaryWriter,
    true_eras: List[int],
    pred_eras: List[int],
    epoch: int,
    num_classes: int = 5,
) -> None:
    """
    Generate and log a confusion matrix for era classification.

    Args:
        writer: TensorBoard SummaryWriter
        true_eras: List of true era labels
        pred_eras: List of predicted era labels
        epoch: Current epoch number
        num_classes: Number of era classes
    """
    try:
        cm = confusion_matrix(true_eras, pred_eras, labels=range(num_classes))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Era")
        plt.ylabel("True Era")
        plt.title("Era Classification Confusion Matrix")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1))
        writer.add_image("validation/era_confusion_matrix", img_tensor, epoch)
        plt.close()
    except Exception as e:
        print(f"Error in log_era_confusion_matrix: {e}")


def log_error_by_era(
    writer: SummaryWriter,
    true_days: List[float],
    pred_days: List[float],
    true_eras: List[int],
    epoch: int,
) -> None:
    """
    Log error metrics broken down by era.

    Args:
        writer: TensorBoard SummaryWriter
        true_days: List of true day values
        pred_days: List of predicted day values
        true_eras: List of true era labels
        epoch: Current epoch number
    """
    try:
        era_errors = {era: [] for era in range(5)}

        for days_true, days_pred, era in zip(true_days, pred_days, true_eras):
            try:
                era = int(era)
                error = abs(float(days_true) - float(days_pred))
                if not np.isnan(error) and not np.isinf(error):
                    era_errors[era].append(error)
            except (ValueError, TypeError):
                continue

        era_mean_errors = {}
        era_names = {
            0: "Primal (65-71)",
            1: "Europe 72-74",
            2: "Hiatus Return (75-79)",
            3: "Brent Era (80-90)",
            4: "Bruce/Vince (90-95)",
        }

        for era, errors in era_errors.items():
            if errors:
                era_mean_errors[era] = sum(errors) / len(errors)
            else:
                era_mean_errors[era] = 0

        for era, mean_error in era_mean_errors.items():
            writer.add_scalar(f"metrics/era_{era}_mae", mean_error, epoch)

        plt.figure(figsize=(10, 6))
        eras_keys = list(era_mean_errors.keys())
        errors = list(era_mean_errors.values())
        x_pos = range(len(eras_keys))

        plt.bar(x_pos, errors)
        plt.xticks(x_pos, [era_names[era] for era in eras_keys])
        plt.xlabel("Era")
        plt.ylabel("Mean Absolute Error (days)")
        plt.title("Dating Error by Era")
        plt.xticks(rotation=45)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = torch.tensor(np.array(img).transpose(2, 0, 1))
        writer.add_image("validation/error_by_era", img_tensor, epoch)
        plt.close()
    except Exception as e:
        print(f"Error in log_error_by_era: {e}")


# Fix missing torch import
import torch
