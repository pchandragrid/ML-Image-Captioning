import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset_384 import COCODataset384
from tqdm import tqdm


def main():

    device = torch.device("mps")
    print("Using device:", device)

    EPOCHS = 5
    BATCH_SIZE = 3  # ⚠️ Lower because 384px uses more memory
    LR = 3e-5

    CHECKPOINT_DIR = "checkpoints_20k_384"
    MODEL_DIR = "saved_model_20k_384"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)

    dataset = COCODataset384(
        "annotations/subset_20k.jsonl",
        "train2017",
        processor
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="mps", dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(MODEL_DIR)
            processor.save_pretrained(MODEL_DIR)
            print("Best model saved.")

        scheduler.step()

    print("Training complete.")


if __name__ == "__main__":
    main()