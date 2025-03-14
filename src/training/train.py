#!/usr/bin/env python3
"""
Training loop implementation for the Grateful Dead show dating model.
"""

import datetime
import os
import platform
import time
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from data.dataset import PreprocessedDeadShowDataset, optimized_collate_fn
from data.preprocessing import preprocess_dataset
from models.dead_model import DeadShowDatingModel
from training.loss import combined_loss
from training.lr_finder import find_learning_rate
from utils.helpers import reset_parameters
from utils.visualization import (
    log_era_confusion_matrix,
    log_error_by_era,
    log_prediction_samples,
)


def train_model(config: Dict) -> None:
    """
    Main training function.

    Args:
        config: Configuration dictionary with training parameters
    """
    # Preprocess dataset
    preprocessed_dir = preprocess_dataset(
        config, force_preprocess=False, store_audio=False
    )

    # Set up logging
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Log configuration
    config_df = pd.DataFrame(list(config.items()), columns=["Parameter", "Value"])
    config_df.to_csv(os.path.join(log_dir, "config.csv"), index=False)
    writer.add_text("Configuration", config_df.to_markdown(), 0)

    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        print(f"Using CUDA. Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            writer.add_text("System", f"GPU {i}: {torch.cuda.get_device_name(i)}", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Set up dataset
    full_dataset = PreprocessedDeadShowDataset(
        preprocessed_dir=preprocessed_dir,
        augment=config["use_augmentation"],
        target_sr=config["target_sr"],
    )

    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(config["valid_split"] * dataset_size)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print(
        f"Total samples: {dataset_size}, Training: {train_size}, Validation: {val_size}"
    )

    # Adjust DataLoader settings based on platform
    is_mac = platform.system() == "Darwin"
    if is_mac:
        print("Running on macOS, adjusting DataLoader settings for compatibility")
        num_workers = 0  # No multiprocessing on Mac
        persistent_workers = False
        dataloader_kwargs = {
            "batch_size": config["batch_size"],
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": True,
            "collate_fn": optimized_collate_fn,
        }
        # Note: No prefetch_factor or persistent_workers when num_workers=0
    else:
        num_workers = min(4, os.cpu_count() - 2)
        persistent_workers = False
        prefetch_factor = 2
        dataloader_kwargs = {
            "batch_size": config["batch_size"],
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": True,
            "collate_fn": optimized_collate_fn,
            "persistent_workers": persistent_workers,
            "prefetch_factor": prefetch_factor,
        }

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    val_dataloader_kwargs = dataloader_kwargs.copy()
    val_dataloader_kwargs["shuffle"] = False
    val_loader = DataLoader(val_dataset, **val_dataloader_kwargs)

    # Initialize model
    model = DeadShowDatingModel(sample_rate=config["target_sr"])
    reset_parameters(model)

    # Apply JIT compilation if configured
    if torch.__version__ >= "1.10.0" and device.type == "cuda" and config["use_jit"]:
        try:
            model = torch.jit.script(model)
            print("Successfully applied JIT compilation to model")
        except Exception as e:
            print(f"JIT compilation failed, using regular model: {e}")

    # Set up multi-GPU if available
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} parameters ({trainable_params:,} trainable)")
    writer.add_text(
        "Model",
        f"Total parameters: {total_params:,} (Trainable: {trainable_params:,})",
        0,
    )

    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["initial_lr"],
        weight_decay=config["weight_decay"],
        eps=1e-8,
    )

    # Run learning rate finder if configured
    if config["run_lr_finder"]:
        print("Running learning rate finder...")
        optimal_lr = find_learning_rate(
            model,
            train_loader,
            optimizer,
            combined_loss,
            device,
            start_lr=1e-6,
            end_lr=1e-1,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = optimal_lr
        config["initial_lr"] = optimal_lr
        print(f"Updated learning rate to {optimal_lr}")

    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["initial_lr"],
        total_steps=config["total_training_steps"],
        pct_start=0.05,
        div_factor=25,
        final_div_factor=1000,
    )

    # Initialize training state
    global_step = 0
    start_epoch = 0
    best_val_mae = float("inf")
    patience_counter = 0

    # Resume from checkpoint if configured
    if config["resume_checkpoint"] and os.path.exists(config["resume_checkpoint"]):
        print(f"Loading checkpoint: {config['resume_checkpoint']}")
        # Add datetime.date to allowed globals for PyTorch 2.6+ compatibility
        try:
            # First try with PyTorch 2.6+ approach using safe globals
            import torch.serialization

            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([datetime.date])
                checkpoint = torch.load(
                    config["resume_checkpoint"], map_location=device
                )
            else:
                # Fall back to older PyTorch versions
                checkpoint = torch.load(
                    config["resume_checkpoint"], map_location=device, weights_only=False
                )
        except (TypeError, AttributeError):
            # Final fallback for even older PyTorch versions
            checkpoint = torch.load(config["resume_checkpoint"], map_location=device)

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        best_val_mae = checkpoint.get("best_val_mae", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)

        print(f"Resumed from step {global_step}, epoch {start_epoch}")

    # Training loop
    epoch = start_epoch
    ema_loss = None
    ema_alpha = 0.98
    training_start_time = time.time()

    print(f"Starting training for {config['total_training_steps']} steps...")

    while global_step < config["total_training_steps"]:
        print(f"Starting epoch {epoch+1}")
        epoch_loss = 0.0
        batch_count = 0
        epoch_mae = 0.0
        epoch_era_correct = 0
        epoch_samples = 0

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
        epoch_start_time = time.time()

        for batch in pbar:
            try:
                # Move batch data to device
                label = batch["label"].to(device, non_blocking=True)
                era = batch["era"].to(device, non_blocking=True)

                # For MPS device, we need to handle audio explicitly
                if "audio" in batch and device.type == "mps":
                    batch["audio"] = batch["audio"].to(device, non_blocking=True)
                elif "raw_audio" in batch and device.type == "mps":
                    batch["raw_audio"] = batch["raw_audio"].to(
                        device, non_blocking=True
                    )

                # Forward pass
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch)

                # Updated: Use combined_loss with global_step and total_steps
                loss = combined_loss(
                    outputs, label, era, global_step, config["total_training_steps"]
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    print("Warning: NaN/Inf detected in loss - skipping batch")
                    continue

                # Backward pass
                loss.backward()

                # Check for valid gradients
                valid_gradients = True
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if (
                            torch.isnan(param.grad).any()
                            or torch.isinf(param.grad).any()
                        ):
                            print(f"NaN/Inf gradient detected in {name}")
                            valid_gradients = False
                            break

                if valid_gradients:
                    # Updated: Use the new gradient clipping value from config
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["grad_clip_value"]
                    )
                    optimizer.step()
                else:
                    print("Skipping optimizer step due to invalid gradients")

                scheduler.step()

                # Update metrics
                current_loss = loss.item()
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * current_loss

                epoch_loss += current_loss
                batch_count += 1
                global_step += 1

                # Calculate batch metrics
                with torch.no_grad():
                    pred_days = outputs["days"].detach()
                    true_days = label.detach()
                    abs_error = torch.abs(pred_days - true_days)
                    batch_mae = torch.mean(abs_error).item()
                    epoch_mae += batch_mae * label.size(0)

                    _, pred_era = torch.max(outputs["era_logits"].detach(), 1)
                    batch_correct = (pred_era == era).sum().item()
                    epoch_era_correct += batch_correct
                    epoch_samples += label.size(0)

                # Log metrics
                writer.add_scalar("loss/train", current_loss, global_step)
                writer.add_scalar("loss/train_ema", ema_loss, global_step)
                writer.add_scalar(
                    "learning_rate", optimizer.param_groups[0]["lr"], global_step
                )

                # Log GPU usage
                if device.type == "cuda" and global_step % 100 == 0:
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
                    memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
                    writer.add_scalar(
                        "system/gpu_memory_allocated_mb", memory_allocated, global_step
                    )
                    writer.add_scalar(
                        "system/gpu_memory_reserved_mb", memory_reserved, global_step
                    )

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss:.4f}",
                        "ema": f"{ema_loss:.4f}",
                        "mae": f"{batch_mae:.1f} days",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                )

                # Check if we've reached the total training steps
                if global_step >= config["total_training_steps"]:
                    break

            except Exception as e:
                print(f"Error in training batch: {e}")
                continue

        # End of epoch
        epoch_time = time.time() - epoch_start_time

        # Calculate epoch metrics
        if batch_count > 0:
            epoch_loss /= batch_count
            epoch_mae /= epoch_samples
            epoch_era_accuracy = 100.0 * epoch_era_correct / epoch_samples

            print(
                f"Epoch {epoch+1} completed in {epoch_time:.1f}s | "
                f"Loss: {epoch_loss:.4f} | MAE: {epoch_mae:.1f} days | "
                f"Era Acc: {epoch_era_accuracy:.2f}%"
            )

            # Log epoch metrics
            writer.add_scalar("metrics/epoch_loss", epoch_loss, epoch)
            writer.add_scalar("metrics/epoch_mae_days", epoch_mae, epoch)
            writer.add_scalar("metrics/epoch_era_accuracy", epoch_era_accuracy, epoch)

        # Validation
        if val_loader:
            (
                val_loss,
                val_mae,
                val_era_acc,
                true_days,
                pred_days,
                true_eras,
                pred_eras,
            ) = validate(model, val_loader, device, config, global_step, writer, epoch)

            print(
                f"Validation | Loss: {val_loss:.4f} | MAE: {val_mae:.1f} days | "
                f"Era Acc: {val_era_acc:.2f}%"
            )

            # Log validation metrics
            writer.add_scalar("metrics/val_loss", val_loss, epoch)
            writer.add_scalar("metrics/val_mae_days", val_mae, epoch)
            writer.add_scalar("metrics/val_era_accuracy", val_era_acc, epoch)

            # Generate validation visualizations
            log_prediction_samples(
                writer, true_days, pred_days, config["base_date"], epoch, prefix="valid"
            )
            log_era_confusion_matrix(writer, true_eras, pred_eras, epoch)
            log_error_by_era(writer, true_days, pred_days, true_eras, epoch)

            # Early stopping
            if config["use_early_stopping"]:
                if val_mae < best_val_mae - config["min_delta"]:
                    best_val_mae = val_mae
                    patience_counter = 0
                    print(f"New best validation MAE: {best_val_mae:.2f} days")

                    # Save best model
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        best_val_mae,
                        patience_counter,
                        os.path.join("checkpoints", config["best_checkpoint"]),
                    )
                else:
                    patience_counter += 1
                    print(
                        f"Validation didn't improve | patience: {patience_counter}/{config['patience']}"
                    )

                    if patience_counter >= config["patience"]:
                        print(
                            f"Early stopping triggered after {patience_counter} epochs without improvement"
                        )
                        break

        # Save checkpoint
        if (epoch + 1) % config["save_every_n_epochs"] == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_val_mae,
                patience_counter,
                os.path.join("checkpoints", config["latest_checkpoint"]),
            )

        epoch += 1

    # End of training
    training_time = time.time() - training_start_time
    print(f"Training completed in {training_time/3600:.2f} hours")

    # Save final model
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        global_step,
        best_val_mae,
        patience_counter,
        os.path.join("checkpoints", "checkpoint_final.pt"),
    )

    writer.close()


def validate(model, val_loader, device, config, global_step, writer, epoch):
    """
    Run validation on the validation set.

    Args:
        model: Model to validate
        val_loader: DataLoader for validation data
        device: Device to run validation on
        config: Configuration dictionary
        global_step: Current global step
        writer: TensorBoard writer
        epoch: Current epoch

    Returns:
        Tuple of (val_loss, val_mae, val_era_acc, true_days, pred_days, true_eras, pred_eras)
    """
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_era_correct = 0
    val_samples = 0

    true_days = []
    pred_days = []
    true_eras = []
    pred_eras = []

    pbar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in pbar:
            # Move batch data to device
            label = batch["label"].to(device, non_blocking=True)
            era = batch["era"].to(device, non_blocking=True)

            # For MPS device, we need to handle audio explicitly
            if "audio" in batch and device.type == "mps":
                batch["audio"] = batch["audio"].to(device, non_blocking=True)
            elif "raw_audio" in batch and device.type == "mps":
                batch["raw_audio"] = batch["raw_audio"].to(device, non_blocking=True)

            # Forward pass
            outputs = model(batch)

            # Updated: Use combined_loss with global_step
            batch_loss = combined_loss(
                outputs, label, era, global_step, config["total_training_steps"]
            )

            # Update metrics
            val_loss += batch_loss.item()

            pred_days_batch = outputs["days"].detach()
            true_days_batch = label.detach()
            abs_error = torch.abs(pred_days_batch - true_days_batch)
            batch_mae = torch.sum(abs_error).item()
            val_mae += batch_mae

            _, pred_era = torch.max(outputs["era_logits"].detach(), 1)
            batch_correct = (pred_era == era).sum().item()
            val_era_correct += batch_correct
            val_samples += label.size(0)

            # Collect data for visualizations
            true_days.extend(true_days_batch.cpu().numpy().tolist())
            pred_days.extend(pred_days_batch.cpu().numpy().tolist())
            true_eras.extend(era.cpu().numpy().tolist())
            pred_eras.extend(pred_era.cpu().numpy().tolist())

            pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

    # Calculate final metrics
    val_loss /= len(val_loader)
    val_mae /= val_samples
    val_era_acc = 100.0 * val_era_correct / val_samples

    return val_loss, val_mae, val_era_acc, true_days, pred_days, true_eras, pred_eras


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_val_mae,
    patience_counter,
    path,
):
    """
    Save a checkpoint of the model.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        epoch: Current epoch
        global_step: Current global step
        best_val_mae: Best validation MAE seen so far
        patience_counter: Current early stopping patience counter
        path: Path to save checkpoint
    """
    state_dict = model.state_dict()
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()

    checkpoint = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_mae": best_val_mae,
        "patience_counter": patience_counter,
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")
