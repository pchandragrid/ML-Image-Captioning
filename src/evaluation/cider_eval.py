import os
from typing import Any

from PIL import Image
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm


def generate_caption(model: Any, processor: Any, image: Image.Image, device) -> str:
    """
    Run the captioning model on a single image and return the decoded caption.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    with getattr(__import__("torch"), "no_grad")():
        torch = __import__("torch")
        generated_ids = model.generate(
            **inputs,
            max_length=30,
            num_beams=5,
        )

    caption = processor.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )
    return caption


def evaluate_cider(model: Any, processor: Any, val_dataset, device, max_samples: int = 200) -> float:
    """
    Compute CIDEr score on a validation subset.

    Expects a PyTorch `Subset`/`Dataset` where:
    - `val_dataset.indices[idx]` gives the underlying index
    - `val_dataset.dataset.annotations[...]` is a list of dicts with
      keys `image` and `captions`.
    """
    import torch  # local import to avoid hard dependency for non-training paths

    model.eval()

    cider_scorer = Cider()
    ground_truth = {}
    predictions = {}

    for idx in tqdm(range(min(max_samples, len(val_dataset))), desc="CIDEr Eval"):
        real_idx = val_dataset.indices[idx]
        ann = val_dataset.dataset.annotations[real_idx]

        image_path = os.path.join("train2017", ann["image"])
        image = Image.open(image_path).convert("RGB")

        pred_caption = generate_caption(model, processor, image, device)

        ground_truth[idx] = ann["captions"]
        predictions[idx] = [pred_caption]

    score, _ = cider_scorer.compute_score(ground_truth, predictions)

    print(f"CIDEr Score: {score:.4f}")

    model.train()
    return score

