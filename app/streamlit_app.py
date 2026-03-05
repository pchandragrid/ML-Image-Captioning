import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
)


@st.cache_resource
def load_caption_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = BlipForConditionalGeneration.from_pretrained("saved_model_phase2")
    processor = BlipProcessor.from_pretrained("saved_model_phase2")

    model.to(device)
    model.eval()

    return model, processor, device


@st.cache_resource
def load_toxicity_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

    model.to(device)
    model.eval()

    return model, tokenizer, device


caption_model, caption_processor, device = load_caption_model()
tox_model, tox_tokenizer, tox_device = load_toxicity_model()


st.title("🖼️ Advanced Image Captioning Demo")
st.write("Fine-tuned BLIP with Beam Search + Toxicity Filtering")

st.sidebar.header("⚙️ Generation Settings")

num_beams = st.sidebar.slider("Beam Size", 1, 10, 5)
max_length = st.sidebar.slider("Max Length", 10, 50, 20)
length_penalty = st.sidebar.slider("Length Penalty", 0.5, 2.0, 1.0, step=0.1)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("Generate Caption"):
        # Generate caption
        with st.spinner("Generating caption..."):
            inputs = caption_processor(
                images=image,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                output_ids = caption_model.generate(
                    **inputs,
                    num_beams=num_beams,
                    max_length=max_length,
                    length_penalty=length_penalty,
                )

            caption = caption_processor.decode(
                output_ids[0],
                skip_special_tokens=True,
            )

        # Confidence score (stable)
        with torch.no_grad():
            loss_inputs = caption_processor(
                images=image,
                text=caption,
                return_tensors="pt",
            ).to(device)

            outputs = caption_model(
                pixel_values=loss_inputs["pixel_values"],
                input_ids=loss_inputs["input_ids"],
                attention_mask=loss_inputs["attention_mask"],
                labels=loss_inputs["input_ids"],
            )

            loss = outputs.loss
            confidence = torch.exp(-loss).item() if loss is not None else 0.0

        # Toxicity check
        tox_inputs = tox_tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
        ).to(tox_device)

        with torch.no_grad():
            tox_outputs = tox_model(**tox_inputs)
            probs = F.softmax(tox_outputs.logits, dim=-1)

        toxic_score = probs[0][1].item()

        # Display caption
        if toxic_score > 0.6:
            st.error("⚠️ Generated caption flagged as potentially toxic.")
            st.markdown("### 🚫 Caption Blocked")
        else:
            st.success("Caption Generated")
            st.markdown(f"### 📝 {caption}")
            st.caption(f"Toxicity Score: {toxic_score:.2f}")
            st.caption(f"Confidence Score: {confidence:.2f}")

        # Vision attention heatmap
        with torch.no_grad():
            vision_outputs = caption_model.vision_model(
                inputs["pixel_values"],
                output_attentions=True,
                return_dict=True,
            )

        attentions = vision_outputs.attentions[-1]
        attn = attentions[0].mean(0)

        cls_attn = attn[0, 1:]
        attn_map = cls_attn.cpu().numpy()
        attn_map = attn_map / attn_map.max()

        size = int(np.sqrt(len(attn_map)))

        fig, ax = plt.subplots()
        ax.imshow(attn_map.reshape(size, size), cmap="viridis")
        ax.set_title("Vision Attention Heatmap")
        ax.axis("off")

        st.pyplot(fig)

