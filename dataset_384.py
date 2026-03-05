import json
import os
import random
from torch.utils.data import Dataset
from PIL import Image


class COCODataset384(Dataset):

    def __init__(self, annotation_path, image_folder, processor):
        self.image_folder = image_folder
        self.processor = processor

        with open(annotation_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        ann = self.annotations[idx]
        caption = random.choice(ann["captions"])

        image_path = os.path.join(self.image_folder, ann["image"])
        image = Image.open(image_path).convert("RGB")

        # 🔥 IMPORTANT: 384px
        image = image.resize((384, 384))

        encoding = self.processor(
            image,
            caption,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": input_ids.clone()
        }