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
    ref_tensor = transform(ref_image).unsqueeze(0).to(device)
    live_tensor = transform(live_image).unsqueeze(0).to(device)

    with torch.no_grad():
        _ = vector_model(ref_tensor)

    with torch.no_grad():
        feat_ref = vector_model(ref_tensor)
        feat_live = vector_model(live_tensor)

        feat_ref = feat_ref.view(feat_ref.size(0), -1)
        feat_live = feat_live.view(feat_live.size(0), -1)

        vector_similarity = F.cosine_similarity(feat_ref, feat_live).item()

        mat_ref = matrix_model(ref_tensor)
        mat_live = matrix_model(live_tensor)

        mat_ref_flat = mat_ref.view(mat_ref.size(0), -1)
        mat_live_flat = mat_live.view(mat_live.size(0), -1)

        matrix_similarity = F.cosine_similarity(mat_ref_flat, mat_live_flat).item()

    final_similarity = 0.7 * vector_similarity + 0.3 * matrix_similarity
    difference = 1.0 - final_similarity
    is_defect = difference >= threshold

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
