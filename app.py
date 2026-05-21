from dataclasses import dataclass
from pathlib import Path
import time

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


@dataclass
class SimilarityResult:
    vector_similarity: float
    matrix_similarity: float
    final_similarity: float
    difference: float
    is_defect: bool


def _remove_transparency(image: Image.Image, bg_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, bg_color)
        background.paste(image, mask=image.split()[3])
        return background

    return image.convert("RGB")


def _load_image(file) -> Image.Image:
    image = Image.open(file)
    return _remove_transparency(image)


@st.cache_resource
def _load_models(device: torch.device) -> tuple[nn.Module, nn.Module]:
    weights_dir = Path.cwd() / "models"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "wide_resnet50_2.pth"
    weights = models.Wide_ResNet50_2_Weights.DEFAULT

    base_model = models.wide_resnet50_2(weights=None)
    try:
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            base_model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError("Local weights not found")
    except Exception:
        state_dict = torch.hub.load_state_dict_from_url(
            weights.url,
            model_dir=str(weights_dir),
            check_hash=True,
            map_location="cpu",
        )
        torch.save(state_dict, weights_path)
        base_model.load_state_dict(state_dict)

    base_model = base_model.to(device)

    vector_model = nn.Sequential(
        *list(base_model.children())[:-1]
    ).to(device)

    matrix_model = nn.Sequential(
        *list(base_model.children())[:-2]
    ).to(device)

    vector_model.eval()
    matrix_model.eval()

    return vector_model, matrix_model


def _get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _compute_similarity(
    ref_image: Image.Image,
    live_image: Image.Image,
    threshold: float,
    device: torch.device,
    vector_model: nn.Module,
    matrix_model: nn.Module,
    transform: transforms.Compose,
) -> SimilarityResult:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Batch inputs
    ref_tensor = transform(ref_image).unsqueeze(0).to(device)
    live_tensor = transform(live_image).unsqueeze(0).to(device)
    batch_tensor = torch.cat([ref_tensor, live_tensor], dim=0)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Single Pass: Matrix Model (batched)
    with torch.no_grad():
        mat_batch = matrix_model(batch_tensor)
        
        # Flatten the matrix outputs to vector form for matrix similarity
        mat_ref_flat = mat_batch[0:1].view(1, -1)
        mat_live_flat = mat_batch[1:2].view(1, -1)
        matrix_similarity = F.cosine_similarity(mat_ref_flat, mat_live_flat).item()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    # Calculate vector features manually from matrix features
    with torch.no_grad():
        feat_batch = F.adaptive_avg_pool2d(mat_batch, (1, 1)).view(2, -1)
        feat_ref = feat_batch[0:1]
        feat_live = feat_batch[1:2]
        vector_similarity = F.cosine_similarity(feat_ref, feat_live).item()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    final_similarity = 0.7 * vector_similarity + 0.3 * matrix_similarity
    difference = 1.0 - final_similarity
    is_defect = difference >= threshold
    
    print(f"--- Inference Time Breakdown ---")
    print(f"Preprocessing + Transfer: {(t1 - t0) * 1000:.2f} ms")
    print(f"Single Batched Forward  : {(t2 - t1) * 1000:.2f} ms")
    print(f"Manual Vector Compute   : {(t3 - t2) * 1000:.2f} ms")
    print(f"Total Similarity Func   : {(t3 - t0) * 1000:.2f} ms")
    print(f"--------------------------------")

    return SimilarityResult(
        vector_similarity=vector_similarity,
        matrix_similarity=matrix_similarity,
        final_similarity=final_similarity,
        difference=difference,
        is_defect=is_defect,
    )


def main() -> None:
    st.set_page_config(page_title="Image Similarity Check", layout="wide")
    st.title("Image Similarity Check")
    st.write("Upload a reference image and a live image to compare their similarity.")

    preview_width = 260

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        st.success(f"GPU configured: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("GPU not available, using CPU")
    vector_model, matrix_model = _load_models(device)
    transform = _get_transform()

    col_left, col_right = st.columns(2)
    with col_left:
        ref_file = st.file_uploader("Reference image", type=["png", "jpg", "jpeg"])
    with col_right:
        live_file = st.file_uploader("Live image", type=["png", "jpg", "jpeg"])

    threshold_pct = st.slider(
        "Defect threshold (difference >= threshold) %",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
    )

    if not ref_file or not live_file:
        st.info("Upload both images to compute similarity.")
        return

    try:
        ref_image = _load_image(ref_file)
        live_image = _load_image(live_file)
    except Exception as exc:
        st.error(f"Failed to read image: {exc}")
        return

    threshold = threshold_pct / 100.0
    start_time = time.perf_counter()
    result = _compute_similarity(
        ref_image,
        live_image,
        threshold,
        device,
        vector_model,
        matrix_model,
        transform,
    )
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    st.subheader("Preview")
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.image(ref_image, caption="Reference image", width=preview_width)
    with preview_col2:
        st.image(live_image, caption="Live image", width=preview_width)

    st.subheader("Results")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Vector similarity", f"{result.vector_similarity * 100:.2f}%")
    with metric_col2:
        st.metric("Matrix similarity", f"{result.matrix_similarity * 100:.2f}%")
    with metric_col3:
        st.metric("Final similarity", f"{result.final_similarity * 100:.2f}%")
    with metric_col4:
        st.metric("Difference (1 - similarity)", f"{result.difference * 100:.2f}%")

    st.metric("Inference time", f"{inference_time_ms:.2f} ms")

    if result.is_defect:
        st.error("Classification: DEFECT")
    else:
        st.success("Classification: OK")


if __name__ == "__main__":
    main()
