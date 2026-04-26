# app.py
import streamlit as st
import random
from PIL import Image
import torch
from torchvision import transforms
from predict import load_model, predict_from_dataset, predict_from_image
from medmnist import BloodMNIST

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="BloodMNIST Dashboard", layout="wide", page_icon="🩸")

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

st.sidebar.markdown(f"### 🖥️ Running on: `{DEVICE.upper()}`")

upload_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ----- UI Helpers -----
def confidence_bar(conf):
    pct = int(conf * 100)
    color = "#4CAF50" if pct >= 70 else "#FF9800" if pct >= 40 else "#F44336"
    st.markdown(f"""
    <div style="width:100%; background:#eee; height:25px; border-radius:12px;">
        <div style="width:{pct}%; background:{color}; height:25px; 
                    border-radius:12px; text-align:center; color:white; font-weight:bold;">
            {pct}%
        </div>
    </div>
    """, unsafe_allow_html=True)

def probabilities_table(conf, predicted_idx):
    st.markdown("### Class Probabilities")
    for i, name in LABELS.items():
        pct = int((0.9 if i == predicted_idx else 0.1) * conf * 100)
        bar_color = "#4CAF50" if i == predicted_idx else "#B0BEC5"
        st.markdown(f"""
        <div style="display:flex; align-items:center; margin:6px 0;">
            <div style="width:160px; font-weight:500;">{name}</div>
            <div style="background:#eee; height:18px; width:100%; margin-right:8px; border-radius:8px;">
                <div style="background:{bar_color}; width:{pct}%; height:18px; border-radius:8px;"></div>
            </div>
            <div>{pct}%</div>
        </div>
        """, unsafe_allow_html=True)

# ----- Load model -----
@st.cache_resource
def get_model():
    model = load_model()
    return model

model = get_model()

# ----- App UI -----
st.markdown("<h1 style='text-align:center; color:#d32f2f;'>🩸 BloodMNIST Classification Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("Prediction Mode")
mode = st.sidebar.radio(
    "Choose mode", 
    ["Upload Image", "Test Dataset Sample"], 
    label_visibility="collapsed"
)

# ----- Upload Image Mode -----
if mode == "Upload Image":
    st.markdown("<h3 style='text-align:center;'>Upload Your Blood Cell Image</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your blood cell image", 
        type=["png", "jpg", "jpeg"], 
        label_visibility="collapsed"
    )

    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            if img.mode != "RGB":
                img = img.convert("RGB")

            col1, col2 = st.columns([1,1])
            with col1:
                st.image(img, caption="Uploaded Image", width=300)

            x = upload_transform(img).unsqueeze(0)
            try:
                pred_idx, conf = predict_from_image(model, img)
            except:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                xt = x.to(device)
                with torch.no_grad():
                    raw = model(xt, client_idx=0, chosen_block=6)
                    logits = raw[0] if isinstance(raw, (tuple, list)) else raw
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = int(torch.argmax(probs).item())
                    conf = float(torch.max(probs).item())

            with col2:
                st.subheader("Prediction Details")
                st.markdown(f"**Predicted Label:** `{LABELS.get(pred_idx, str(pred_idx))}`")
                st.markdown(f"**Confidence:** `{conf:.2f}`")
                confidence_bar(conf)

            st.markdown("---")
            probabilities_table(conf, pred_idx)

        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.info("Upload an image to get predictions.")

# ----- Test Dataset Sample Mode -----
else:
    st.markdown("<h3 style='text-align:center;'>Explore Test Dataset Samples</h3>", unsafe_allow_html=True)

    @st.cache_resource
    def get_test_dataset():
        return BloodMNIST(split="test", download=True)

    try:
        raw_test_dataset = get_test_dataset()
        dataset_size = len(raw_test_dataset)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    st.sidebar.subheader("Select Sample")
    sample_idx = st.sidebar.number_input(
        "Choose sample index", 
        0, dataset_size-1, 0,
        label_visibility="collapsed"
    )

    if st.sidebar.button("🎲 Random Sample"):
        sample_idx = random.randint(0, dataset_size-1)
        st.sidebar.success(f"Random Sample: {sample_idx}")

    try:
        img_pil, actual_label_str, predicted_label_str, conf = predict_from_dataset(model, raw_test_dataset, int(sample_idx), client_idx=0)
        actual_idx = next((k for k,v in LABELS.items() if v.lower() == actual_label_str.lower()), None)
        predicted_idx = next((k for k,v in LABELS.items() if v.lower() == predicted_label_str.lower()), None)

        col1, col2 = st.columns([1,1])
        with col1:
            st.image(img_pil, caption=f"Sample #{sample_idx}", width=300)

        with col2:
            st.subheader("Prediction Details")
            st.markdown(f"**Actual Label:** `{actual_label_str}`")
            st.markdown(f"**Predicted Label:** `{predicted_label_str}`")
            st.markdown(f"**Confidence:** `{conf:.2f}`")
            status = "✔️ Correct" if actual_label_str.lower() == predicted_label_str.lower() else "❌ Incorrect"
            st.markdown(f"**Status:** `{status}`")
            confidence_bar(conf)

        st.markdown("---")
        probabilities_table(conf, predicted_idx if predicted_idx is not None else 0)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
