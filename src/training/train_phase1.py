import os

import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from transformers import BlipForConditionalGeneration, BlipProcessor
from tqdm import tqdm

from src.data.coco_384_dataset import COCODataset384 as COCODataset
from src.evaluation.cider_eval import evaluate_cider


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available.")

    device = torch.device("mps")
    print("Using device:", device)

    # =========================
    # CONFIG
    # =========================
    EPOCHS = 5
    BATCH_SIZE = 6
    LR = 3e-5
    NUM_WORKERS = 0
    FINAL_MODEL_DIR = "saved_model_phase1"

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

    # =========================
    # LOAD MODEL
    # =========================
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    # Unfreeze LAST 2 vision layers only
    for name, param in model.vision_model.named_parameters():
        if "encoder.layers.10" in name or "encoder.layers.11" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)

    # =========================
    # DATASET SPLIT
    # =========================
    full_dataset = COCODataset(
        "annotations/subset_10k.jsonl",
        "train2017",
        processor,
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # =========================
    # EARLY STOPPING
    # =========================
    best_cider = 0.0
    patience = 3
    counter = 0

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="mps", dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}")

        # =========================
        # VALIDATION LOSS
        # =========================
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")

        # =========================
        # CIDEr
        # =========================
        cider_score = evaluate_cider(model, processor, val_dataset, device)

        # =========================
        # SAVE BEST CIDEr MODEL
        # =========================
        if cider_score > best_cider:
            best_cider = cider_score
            counter = 0
            model.save_pretrained(FINAL_MODEL_DIR)
            processor.save_pretrained(FINAL_MODEL_DIR)
            print("Best CIDEr model saved.")
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered.")
            break

        scheduler.step()

    print("Phase 1 training complete.")


if __name__ == "__main__":
    main()

