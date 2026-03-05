import json
import matplotlib.pyplot as plt
import numpy as np

ANNOTATION_FILE = "annotations/captions_validation.jsonl"

short = []
medium = []
long = []

with open(ANNOTATION_FILE) as f:
    for line in f:
        data = json.loads(line)

        caption = data["captions"][0]
        length = len(caption.split())

        if length <= 8:
            short.append(length)

        elif length <= 15:
            medium.append(length)

        else:
            long.append(length)

print("Short captions:", len(short))
print("Medium captions:", len(medium))
print("Long captions:", len(long))


# Example scores from your training logs
blip_scores = [0.71, 0.60, 0.48]
vit_scores = [0.65, 0.59, 0.42]
git_scores = [0.30, 0.18, 0.11]

labels = ["Short", "Medium", "Long"]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(9,5))

plt.bar(x - width, blip_scores, width, label="BLIP")
plt.bar(x, vit_scores, width, label="ViT-GPT2")
plt.bar(x + width, git_scores, width, label="GIT")

plt.xlabel("Caption Length")
plt.ylabel("CIDEr Score")
plt.title("Model Performance vs Caption Length")

plt.xticks(x, labels)

plt.legend()

plt.savefig("caption_length_analysis.png", dpi=300)

plt.show()