import json
import os
import random
from typing import Any, Dict, List

from PIL import Image
from torch.utils.data import Dataset


class COCODatasetViTGPT2(Dataset):
    """
    COCO dataset tailored for ViT + GPT-2 style architectures with
    separate image processor and tokenizer.
    """

    def __init__(
        self,
        annotation_path: str,
        image_folder: str,
        image_processor: Any,
        tokenizer: Any,
        mode: str = "short",
        max_length: int = 20,
    ) -> None:
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        with open(annotation_path, "r") as f:
            raw_data = [json.loads(line) for line in f]

        self.annotations: List[Dict[str, Any]] = []

        for ann in raw_data:
            filtered: List[str] = []

            for cap in ann["captions"]:
                words = cap.split()
                wc = len(words)

                if mode == "short" and wc <= 8:
                    filtered.append(cap)
                elif mode == "long" and wc > 15:
                    filtered.append(cap)
                elif mode == "mixed":
                    filtered.append(cap)

            if filtered:
                self.annotations.append(
                    {
                        "image": ann["image"],
                        "captions": filtered,
                    }
                )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        caption = random.choice(ann["captions"])

        image_path = os.path.join(self.image_folder, ann["image"])
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.squeeze(0)

        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": input_ids,
        }

