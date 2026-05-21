from dataclasses import dataclass
from pathlib import Path
import time
import io
import hashlib
import numpy as np

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


def _compute_features_batch(images: list[Image.Image], device: torch.device, feature_model: nn.Module, transform: transforms.Compose) -> tuple[np.ndarray, np.ndarray]:
    tensors = [transform(img) for img in images]
    batch_tensor = torch.stack(tensors).to(device)
    
    with torch.no_grad():
        feat = feature_model(batch_tensor)

    pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)

    pooled_np = pooled.cpu().numpy()
    mat_flat_np = feat.view(feat.size(0), -1).cpu().numpy()
    return pooled_np, mat_flat_np


@st.cache_resource
def _load_models(device: torch.device) -> nn.Module:
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

    feature_model = nn.Sequential(
        *list(base_model.children())[:-2]
    ).to(device)

    feature_model.eval()

    return feature_model


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
    feature_model: nn.Module,
    transform: transforms.Compose,
) -> SimilarityResult:
    # compute features for both images (may be cached externally)
    pooled_ref, mat_ref_flat = _compute_features(ref_image, device, feature_model, transform)
    pooled_live, mat_live_flat = _compute_features(live_image, device, feature_model, transform)

    # cosine similarity on numpy arrays
    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)
        denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a_flat, b_flat) / denom)

    vector_similarity = cos_sim(pooled_ref, pooled_live)
    matrix_similarity = cos_sim(mat_ref_flat, mat_live_flat)

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
    feature_model = _load_models(device)
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
        # read raw bytes to create a reproducible cache key
        ref_bytes = ref_file.read()
        ref_file.seek(0)
        live_bytes = live_file.read()
        live_file.seek(0)

        ref_key = hashlib.md5(ref_bytes).hexdigest()

        ref_image = Image.open(io.BytesIO(ref_bytes))
        ref_image = _remove_transparency(ref_image)
        live_image = Image.open(io.BytesIO(live_bytes))
        live_image = _remove_transparency(live_image)
    except Exception as exc:
        st.error(f"Failed to read image: {exc}")
        return

    threshold = threshold_pct / 100.0

    # check cache for reference features
    cached = False
    if "ref_cache" in st.session_state and st.session_state.ref_cache.get("key") == ref_key:
        pooled_ref = st.session_state.ref_cache["pooled"]
        mat_ref_flat = st.session_state.ref_cache["mat_flat"]
        cached = True
    
    # warm GPU and measure inference
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    if cached:
        pooled_live_batch, mat_live_flat_batch = _compute_features_batch([live_image], device, feature_model, transform)
        pooled_live = pooled_live_batch[0]
        mat_live_flat = mat_live_flat_batch[0]
    else:
        # compute both in a single batch
        pooled_batch, mat_flat_batch = _compute_features_batch([ref_image, live_image], device, feature_model, transform)
        pooled_ref, pooled_live = pooled_batch[0], pooled_batch[1]
        mat_ref_flat, mat_live_flat = mat_flat_batch[0], mat_flat_batch[1]
        
        # update cache
        st.session_state.ref_cache = {"key": ref_key, "pooled": pooled_ref, "mat_flat": mat_ref_flat}

    # compute similarities using numpy arrays
    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)
        denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a_flat, b_flat) / denom)

    vector_similarity = cos_sim(pooled_ref, pooled_live)
    matrix_similarity = cos_sim(mat_ref_flat, mat_live_flat)
    final_similarity = 0.7 * vector_similarity + 0.3 * matrix_similarity
    difference = 1.0 - final_similarity
    is_defect = difference >= threshold

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_ms = (time.perf_counter() - start_time) * 1000

    result = SimilarityResult(
        vector_similarity=vector_similarity,
        matrix_similarity=matrix_similarity,
        final_similarity=final_similarity,
        difference=difference,
        is_defect=is_defect,
    )

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
