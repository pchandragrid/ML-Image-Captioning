import json
import os
import random
from typing import Any, Dict

from PIL import Image
from torch.utils.data import Dataset


class COCODataset384(Dataset):
    """
    COCO-style dataset that always resizes images to 384x384 and uses
    a BLIP-style processor for joint image-text encoding.
    """

    def __init__(self, annotation_path: str, image_folder: str, processor: Any) -> None:
        self.image_folder = image_folder
        self.processor = processor

        with open(annotation_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        caption = random.choice(ann["captions"])

        image_path = os.path.join(self.image_folder, ann["image"])
        image = Image.open(image_path).convert("RGB")

        # 384px resize for the vision backbone
        image = image.resize((384, 384))

        encoding = self.processor(
            image,
            caption,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": input_ids.clone(),
        }

