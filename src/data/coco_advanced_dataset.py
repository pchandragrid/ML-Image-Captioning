import json
import os
import random
import re
from typing import Any, Dict, List

from PIL import Image
from torch.utils.data import Dataset


class COCODatasetAdvanced(Dataset):
    """
    COCO dataset with caption quality and length filtering.
    """

    def __init__(
        self,
        annotation_path: str,
        image_folder: str,
        processor: Any,
        mode: str = "mixed",
        max_length: int = 40,
    ) -> None:
        self.image_folder = image_folder
        self.processor = processor
        self.max_length = max_length
        self.mode = mode

        with open(annotation_path, "r") as f:
            raw_data = [json.loads(line) for line in f]

        self.annotations: List[Dict[str, Any]] = []

        for ann in raw_data:
            filtered_captions: List[str] = []

            for cap in ann["captions"]:
                cap = cap.strip().lower()

                # Remove very short captions
                if len(cap.split()) < 3:
                    continue

                # Remove repeated words
                words = cap.split()
                if len(set(words)) < len(words) * 0.6:
                    continue

                # Remove non-alphabetic captions
                if not re.search(r"[a-z]", cap):
                    continue

                word_count = len(words)

                if self.mode == "short" and word_count <= 8:
                    filtered_captions.append(cap)
                elif self.mode == "long" and word_count > 15:
                    filtered_captions.append(cap)
                elif self.mode == "mixed":
                    filtered_captions.append(cap)

            if filtered_captions:
                self.annotations.append(
                    {
                        "image": ann["image"],
                        "captions": filtered_captions,
                    }
                )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        file_name = ann["image"]
        caption = random.choice(ann["captions"])

        image_path = os.path.join(self.image_folder, file_name)
        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": input_ids.clone(),
        }

