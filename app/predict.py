import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from medmnist import BloodMNIST

from models import FeSVBiS

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_PATH = "final_state_dict.pth"

class_names = [
    'basophil', 'eosinophil', 'erythroblast',
    'immature granulocyte', 'lymphocyte',
    'monocyte', 'neutrophil', 'platelet'
]


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")



def load_model():
    print("Loading FeSVBiS model...")

    # 1. Create model on CPU (safest for loading)
    model = FeSVBiS(
        ViT_name="vit_base_r50_s16_224",
        num_classes=8,
        num_clients=6,
        in_channels=3,
        ViT_pretrained=False,
        initial_block=1,
        final_block=6,
        resnet_dropout=0.5,
        DP=False,
        mean=0,
        std=0
    )
    model.cpu()

    # 2. Load checkpoint on CPU
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # 3. Fix tokens safely
    fixed_sd = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.clone()

        if "cls_token" in k and isinstance(v, torch.Tensor) and v.ndim == 1:
            embed = v.shape[0]
            print(f"Fix cls_token → (1,1,{embed})")
            v = v.view(1, 1, embed)

        if "pos_embed" in k and isinstance(v, torch.Tensor):
            if v.ndim == 1:
                embed_dim = v.shape[0]
                model_param = dict(model.named_parameters()).get(k, None)
                if model_param is not None:
                    _, num_tokens, embed_dim = model_param.shape
                else:
                    num_tokens = 197
                print(f"Fix pos_embed → (1,{num_tokens},{embed_dim})")
                v = v.view(1, num_tokens, embed_dim)

            elif v.ndim == 2:
                v = v.unsqueeze(0)

        fixed_sd[k] = v

    # 4. Load into model
    try:
        model.load_state_dict(fixed_sd, strict=True)
        print("Loaded with strict=True")
    except Exception as e:
        print("Strict load failed → fallback strict=False")
        model.load_state_dict(fixed_sd, strict=False)

    # 5. Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 6. Move model to GPU if available
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = model.to(device)
        print(f"Model moved to {device}")
    except Exception as e:
        print("GPU move failed, staying on CPU:", e)
        device = torch.device("cpu")
        model = model.to(device)

    model.device = device
    model.eval()

    print("Model ready.")
    return model


# ---------------------------------------------------------
# Transform (ViT expects 224×224)
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# ---------------------------------------------------------
# Extract model output safely
# ---------------------------------------------------------
def unwrap_output(ret):
    if isinstance(ret, torch.Tensor):
        return ret
    if isinstance(ret, (tuple, list)):
        return ret[0]
    return torch.tensor(ret)


# ---------------------------------------------------------
# Predict from dataset
# ---------------------------------------------------------
def predict_from_dataset(model, dataset, idx, client_idx=0):
    img_pil, label = dataset[idx]

    device = _model_device(model)
    img_input = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        raw = model(img_input, client_idx=client_idx, chosen_block=6)
        logits = unwrap_output(raw)
        probs = torch.softmax(logits, dim=1)
        conf, cls = probs.max(1)

    predicted_class = class_names[cls.item()]
    actual_class = class_names[int(label)]

    return img_pil, actual_class, predicted_class, float(conf.item())


# ---------------------------------------------------------
# Predict from PIL image
# ---------------------------------------------------------
def predict_from_image(model, img: Image.Image):
    try:
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        device = _model_device(model)
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            raw = model(x, client_idx=0, chosen_block=6)
            logits = unwrap_output(raw)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            conf = float(torch.max(probs).item())

        return pred_idx, conf

    except Exception as e:
        raise RuntimeError(f"Failed to predict image: {e}")



# ---------------------------------------------------------
# Evaluate Dataset Accuracy (GPU-optimized)
# ---------------------------------------------------------
def evaluate_accuracy(model, dataset):
    device = _model_device(model)
    total = len(dataset)
    correct = 0

    print(f"\nEvaluating accuracy on {total} samples...")

    for i in range(total):
        img_pil, label = dataset[i]
        x = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = unwrap_output(model(x, client_idx=0, chosen_block=6))
            cls = pred.argmax(1).item()

        if cls == int(label):
            correct += 1

        if i % 500 == 0:
            print(f"{i}/{total} samples...")

    acc = correct / total * 100
    print(f"\nAccuracy: {acc:.2f}%")
    return acc


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    model = load_model()
    dataset = BloodMNIST(split="test", download=True)

    # Test single sample
    idx = 1
    img, actual, pred, conf = predict_from_dataset(model, dataset, idx)

    plt.imshow(img)
    plt.title(f"Actual: {actual}")
    plt.axis("off")
    plt.savefig(f"sample_{idx}.png")
    plt.close()

    print("\nRESULT")
    print("------")
    print("Actual:", actual)
    print("Predicted:", pred)
    print("Confidence:", conf)

    # Full dataset accuracy
    evaluate_accuracy(model, dataset)