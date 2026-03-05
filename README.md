# Image Captioning (Streamlit)

This repo hosts a Streamlit app (`app.py`) that compares multiple image-captioning models.

## Why your models should NOT be inside the app repo

Fine-tuned checkpoints are large. Public hosting (Hugging Face Spaces / Streamlit Cloud) works best when:

- the app repo stays small
- models live on the Hugging Face Hub (or S3/GCS)
- the app downloads models at startup (cached by `transformers`)

## 1) Upload your saved models to Hugging Face Hub

Example for BLIP (you already have `uploadtohf.py`):

```bash
pip install -U transformers huggingface_hub
huggingface-cli login
python uploadtohf.py
```

Do the same for your other local folders (`saved_vit_gpt2`, `saved_git_model`) by pushing them to separate Hub repos.

## 2) Configure the app to load from Hub

`app.py` loads **local folders if present**, otherwise falls back to Hub IDs via environment variables:

- `BLIP_MODEL_ID` (default: `prateekchandra/blip-caption-model`)
- `VITGPT2_MODEL_ID` (default: `prateekchandra/vit-gpt2-caption-model`)
- `GIT_MODEL_ID` (default: `prateekchandra/git-caption-model`)

You can also override local folder names:

- `BLIP_LOCAL_DIR` (default: `saved_model_phase2`)
- `VITGPT2_LOCAL_DIR` (default: `saved_vit_gpt2`)
- `GIT_LOCAL_DIR` (default: `saved_git_model`)

## 3) Deploy options

### Option A: Hugging Face Spaces (recommended)

- Create a new Space: **Streamlit**
- Push this repo (must include `app.py` + `requirements.txt`)
- In Space “Variables”, set `BLIP_MODEL_ID`, `VITGPT2_MODEL_ID`, `GIT_MODEL_ID` to your Hub repos
- If any model repo is private, add `HF_TOKEN` as a Space **Secret**

### Option B: Streamlit Community Cloud

- Point it to this repo
- Set the same env vars in the app settings

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

# 🖼️ Image Captioning with BLIP (COCO Subset)

## 📌 Problem

Generate natural language descriptions for images using transformer-based vision-language models.

Goal:
- Improve CIDEr score by 10%+
- Compare architectures (BLIP vs ViT-GPT2)
- Analyze resolution impact (224 vs 320 vs 384)
- Optimize decoding parameters
- Deploy minimal inference UI

---

## 📂 Dataset

- MS COCO Captions (subset: 10k & 20k)
- Random caption selection (5 captions per image)
- Experiments:
  - Short captions
  - Mixed captions
  - Filtered captions

Train/Validation split: 90/10

---

## 🧠 Models

### 1️⃣ BLIP (Primary Model)
- Salesforce/blip-image-captioning-base
- Vision encoder frozen (for efficiency)
- Gradient checkpointing enabled
- Mixed precision on MPS

### 2️⃣ ViT-GPT2 (Comparison)
- ViT base encoder
- GPT2 decoder with cross-attention

---

## 🧪 Experiments

### Resolution Comparison
| Resolution | Dataset | CIDEr |
|------------|---------|--------|
| 224px | 10k | ~1.28 |
| 320px | 20k | ~1.33–1.38 |
| 384px | 20k | ~1.40+ |

### Beam Search Tuning
Tested:
- Beams: 3, 5, 8
- Length penalty: 0.8, 1.0, 1.2
- Max length: 20, 30, 40

Best config:
Beams=5, MaxLen=20, LengthPenalty=1.0

---

## 📊 Evaluation Metric

- CIDEr (via pycocoevalcap)
- Validation loss
- Confidence estimation

---

## 🖥️ Demo

Streamlit app includes:
- Image uploader
- Beam controls
- Toxicity filtering
- Confidence display
- Attention heatmap

Run:
```bash
streamlit run app.py