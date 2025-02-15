import datetime
import os

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModel,
    Wav2Vec2FeatureExtractor,
    get_linear_schedule_with_warmup,
)

# --- configuration ---
input_dir = "../../data/audsnippets-all"  # directory structure: year/date (e.g., "1969", then files)
target_sr = 24000  # already at target sample rate; no resampling
batch_size = 32
initial_lr = 1e-4
num_workers = 2
valid_split = 0.1
base_year = 1968
base_date = datetime.date(
    base_year, 1, 1
)  # for converting dates to regression target (days since)
resume_checkpoint = "checkpoint_latest.pt"  # checkpoint file for resume (if exists)
checkpoint_freq = 1000  # save checkpoint & validate every 1000 steps

# for an indefinite training loop, we set the total training steps in the scheduler to a HUGE number
scheduler_total_steps = 10**9


# --- dataset definition ---
class snippetdataset(Dataset):
    def __init__(self, root_dir, base_date, target_sr=24000):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.base_date = base_date
        self.files = []
        self.labels = []
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.startswith(f"{subdir}-00-00"):
                        continue
                    # use the first 10 chars (yyyy-mm-dd) as the date
                    date_str = file[:10]
                    try:
                        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    except Exception as e:
                        print(f"skipping {file} due to date parse error: {e}")
                        continue
                    days = (date_obj.date() - self.base_date).days
                    self.files.append(os.path.join(subdir_path, file))
                    self.labels.append(days)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        try:
            # files are already at target_sr; use sr=None
            y, _ = librosa.load(file_path, sr=None)
        except Exception as e:
            print(f"error loading {file_path}: {e}")
            y = np.zeros(int(self.target_sr * 15))  # 15s of silence on failure
        return {
            "audio": y,
            "label": torch.tensor(label, dtype=torch.float),
            "file": file_path,
        }


def collate_fn(batch):
    audios = [torch.tensor(item["audio"]) for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    audios = torch.stack(audios)
    return {"audio": audios, "label": labels}


# --- regression head definition ---
class regressionhead(nn.Module):
    def __init__(self, input_dim):
        super(regressionhead, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def main():
    writer = SummaryWriter(log_dir="logs")
    global_step = 0
    start_epoch = 0

    # prepare dataset and data loaders
    full_dataset = snippetdataset(input_dir, base_date, target_sr=target_sr)
    dataset_size = len(full_dataset)
    val_size = int(valid_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(
        f"total samples: {dataset_size}, training: {train_size}, validation: {val_size}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # try lowering or even 0 if needed
        collate_fn=collate_fn,
        # removed persistent_workers & multiprocessing_context
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        # removed persistent_workers & multiprocessing_context
    )

    # load processor and base model
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "m-a-p/MERT-v1-330M", trust_remote_code=True
    )
    base_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    hidden_dim = base_model.config.hidden_size
    reg_head = regressionhead(hidden_dim)

    # device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    base_model.to(device)
    reg_head.to(device)

    # full model training: no freezing
    optimizer = optim.AdamW(
        list(base_model.parameters()) + list(reg_head.parameters()),
        lr=initial_lr,
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=scheduler_total_steps
    )
    criterion = nn.MSELoss()

    use_amp = False
    if device.type == "mps":
        try:
            with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                pass
            use_amp = True
            print("mixed precision enabled on mps using bfloat16.")
        except Exception as e:
            print(
                "mixed precision autocast not supported on mps; proceeding in fp32.", e
            )

    # resume from checkpoint if available
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        base_model.load_state_dict(ckpt["base_model_state_dict"])
        reg_head.load_state_dict(ckpt["regression_head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        print(
            f"resumed from {resume_checkpoint} at global step {global_step}, epoch {start_epoch+1}"
        )

    epoch = start_epoch
    while True:
        print(f"starting epoch {epoch+1}")
        epoch_loss = 0.0
        batch_count = 0
        # iterate through train loader
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1} training"):
            batch_count += 1
            global_step += 1

            inputs = processor(
                batch["audio"].tolist(),
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True,
            )
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16 if device.type == "mps" else torch.float16,
                ):
                    outputs = base_model(**inputs)
                    hidden_states = outputs.last_hidden_state
                    pooled = torch.mean(hidden_states, dim=1)
                    preds = reg_head(pooled).squeeze(-1)
                    loss = criterion(preds, labels)
                loss.backward()
            else:
                outputs = base_model(**inputs)
                hidden_states = outputs.last_hidden_state
                pooled = torch.mean(hidden_states, dim=1)
                preds = reg_head(pooled).squeeze(-1)
                loss = criterion(preds, labels)
                loss.backward()

            optimizer.step()
            scheduler.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            writer.add_scalar("loss/iter_loss", current_loss, global_step)
            writer.add_scalar(
                "learning_rate", optimizer.param_groups[0]["lr"], global_step
            )

            # frequent checkpointing & validation
            if global_step % checkpoint_freq == 0:
                # validation
                base_model.eval()
                reg_head.eval()
                total_val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for batch_val in val_loader:
                        inputs_val = processor(
                            batch_val["audio"].tolist(),
                            sampling_rate=target_sr,
                            return_tensors="pt",
                            padding=True,
                        )
                        for k, v in inputs_val.items():
                            inputs_val[k] = v.to(device)
                        labels_val = batch_val["label"].to(device)
                        if use_amp:
                            with torch.autocast(
                                device_type=device.type,
                                dtype=(
                                    torch.bfloat16
                                    if device.type == "mps"
                                    else torch.float16
                                ),
                            ):
                                outputs_val = base_model(**inputs_val)
                                hidden_states_val = outputs_val.last_hidden_state
                                pooled_val = torch.mean(hidden_states_val, dim=1)
                                preds_val = reg_head(pooled_val).squeeze(-1)
                                loss_val = criterion(preds_val, labels_val)
                        else:
                            outputs_val = base_model(**inputs_val)
                            hidden_states_val = outputs_val.last_hidden_state
                            pooled_val = torch.mean(hidden_states_val, dim=1)
                            preds_val = reg_head(pooled_val).squeeze(-1)
                            loss_val = criterion(preds_val, labels_val)
                        total_val_loss += loss_val.item()
                        val_batches += 1
                avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0.0
                print(
                    f"\nvalidation at step {global_step}: avg loss = {avg_val_loss:.4f}"
                )
                writer.add_scalar("val_loss", avg_val_loss, global_step)
                base_model.train()
                reg_head.train()

                # save checkpoint
                ckpt_data = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "base_model_state_dict": base_model.state_dict(),
                    "regression_head_state_dict": reg_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                ckpt_path = f"checkpoint_step_{global_step}.pt"
                torch.save(ckpt_data, ckpt_path)
                torch.save(ckpt_data, resume_checkpoint)
                print(f"saved checkpoint at step {global_step}")

        avg_epoch_loss = epoch_loss / batch_count if batch_count else 0.0
        print(f"epoch {epoch+1} complete: average training loss = {avg_epoch_loss:.4f}")
        writer.add_scalar("epoch/train_loss", avg_epoch_loss, epoch + 1)
        epoch += 1

    # never reached (unless training is stopped externally)
    writer.close()


if __name__ == "__main__":
    main()
