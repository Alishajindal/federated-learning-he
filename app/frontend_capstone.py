import streamlit as st
import numpy as np
import random
from predict import load_model, predict_from_dataset
from medmnist import BloodMNIST
from PIL import Image
import torch
from torchvision import transforms

# =====================================================
# CONFIG — Auto GPU if available
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# Streamlit Config
# =====================================================
st.set_page_config(page_title="BloodMNIST Dashboard", layout="wide")

# =====================================================
# Load model & dataset (cached)
# =====================================================
@st.cache_resource
def load_resources():
    model = load_model()  # load from predict.py
    model.to(DEVICE)      # send model to GPU if available
    dataset = BloodMNIST(split="test", download=True)
    return model, dataset

model, raw_test_dataset = load_resources()
dataset_size = len(raw_test_dataset)

LABELS = {
    0: "Basophil",
    1: "Eosinophil",
    2: "Erythroblast",
    3: "Immature Granulocyte",
    4: "Lymphocyte",
    5: "Monocyte",
    6: "Neutrophil",
    7: "Platelet"
}

# Show device info
st.sidebar.markdown(f"### 🖥️ Running on: `{DEVICE.upper()}`")

# =====================================================
# Preprocessing
# =====================================================
upload_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =====================================================
# Utilities
# =====================================================
def to_index(x):
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        for k, v in LABELS.items():
            if v.lower() == x.lower().strip():
                return k
    raise ValueError(f"Cannot convert label: {x}")

def confidence_bar(conf):
    pct = int(conf * 100)
    color = "#4CAF50" if pct >= 70 else "#FF9800" if pct >= 40 else "#F44336"

    st.markdown(f"""
    <div style="width:100%; background:#ccc; height:25px; border-radius:8px;">
        <div style="width:{pct}%; background:{color}; height:25px; 
                    border-radius:8px; text-align:center; color:white; font-weight:bold;">
            {pct}%
        </div>
    </div>
    """, unsafe_allow_html=True)

def probabilities_table(conf, predicted_idx):
    st.markdown("### Class Probabilities")
    for i, name in LABELS.items():
        pct = int((0.9 if i == predicted_idx else 0.1) * conf * 100)
        bar_color = "#4CAF50" if i == predicted_idx else "#9E9E9E"

        st.markdown(f"""
        <div style="display:flex; align-items:center; margin:6px 0;">
            <div style="width:140px;">{name}</div>
            <div style="background:#eee; height:18px; width:100%; margin-right:8px;">
                <div style="background:{bar_color}; width:{pct}%; height:18px;"></div>
            </div>
            <div>{pct}%</div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# Prediction Functions
# =====================================================
def predict_sample(idx):
    img, actual, predicted, conf = predict_from_dataset(
        model,
        raw_test_dataset,
        int(idx),
        client_idx=0
    )
    actual_idx = to_index(actual)
    predicted_idx = to_index(predicted)
    return img, actual_idx, predicted_idx, conf


def predict_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")

    x = upload_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        raw = model(x, client_idx=0, chosen_block=6)
        logits = raw if isinstance(raw, torch.Tensor) else raw[0]
        probs = torch.softmax(logits, dim=1)

        predicted_idx = int(torch.argmax(probs, dim=1).item())
        conf = float(torch.max(probs).item())

    return img, predicted_idx, conf

# =====================================================
# Streamlit UI
# =====================================================
st.markdown("<h1 style='text-align:center;'>BloodMNIST Classification Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Choose Prediction Mode")
mode = st.sidebar.radio("Select Mode", ["Test Dataset Sample", "Upload Image"], key="mode_select")

# =====================================================
# MODE 1 — Test Sample
# =====================================================
if mode == "Test Dataset Sample":
    st.markdown("<h3 style='text-align:center;'>Use test samples</h3>", unsafe_allow_html=True)

    st.sidebar.subheader("🔎︎ Choose a Sample")

    sample_idx = st.sidebar.number_input(
        "Sample Index",
        min_value=0,
        max_value=dataset_size - 1,
        step=1,
        value=None,
        placeholder="Enter an index",
        key="sample_index_test_mode"
    )

    if st.sidebar.button("🎲 Random Sample"):
        sample_idx = random.randint(0, dataset_size - 1)
        st.sidebar.success(f"Random Sample Selected: {sample_idx}")

    if sample_idx is not None:
        img, actual_idx, predicted_idx, conf = predict_sample(sample_idx)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption=f"Sample #{sample_idx}", width=300)

        with col2:
            st.subheader("Prediction Details")
            st.write(f"**Actual Label:** {LABELS[actual_idx]}")
            st.write(f"**Predicted Label:** {LABELS[predicted_idx]}")
            st.write(f"**Confidence:** {conf:.4f}")

            status = "✔️ Correct" if actual_idx == predicted_idx else "❌ Incorrect"
            st.write(f"**Status:** {status}")

            confidence_bar(conf)

        st.divider()
        probabilities_table(conf, predicted_idx)

    else:
        st.info("Enter a sample index or Upload your Own Image or click Random.")

# =====================================================
# MODE 2 — Upload Image
# =====================================================
elif mode == "Upload Image":

    st.markdown("<h3 style='text-align:center;'>Upload your own image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a blood cell image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img, predicted_idx, conf = predict_uploaded_image(uploaded_file)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Uploaded Image", width=300)

        with col2:
            st.subheader("Prediction Details")
            st.write(f"**Predicted Label:** {LABELS[predicted_idx]}")
            st.write(f"**Confidence:** {conf:.4f}")
            confidence_bar(conf)

        st.divider()
        probabilities_table(conf, predicted_idx)

    else:
        st.info("Upload an image to begin.")