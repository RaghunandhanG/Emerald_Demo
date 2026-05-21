from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# -----------------------------------
# REMOVE TRANSPARENCY
# -----------------------------------
def remove_transparency(img, bg_color=(255, 255, 255)):

    if img.mode == 'RGBA':
        background = Image.new("RGB", img.size, bg_color)
        background.paste(img, mask=img.split()[3])
        return background

    return img.convert("RGB")


# -----------------------------------
# DEVICE
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device: {device}")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# -----------------------------------
# LOAD MODEL
# Downloads only once automatically
# Then uses local cache forever
# -----------------------------------
base_model = models.wide_resnet50_2(pretrained=True).to(device)

# 1D feature extractor
vector_model = nn.Sequential(
    *list(base_model.children())[:-1]
).to(device)

# Matrix feature extractor
matrix_model = nn.Sequential(
    *list(base_model.children())[:-2]
).to(device)

vector_model.eval()
matrix_model.eval()


# -----------------------------------
# IMAGE PREPROCESSING
# -----------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------------
# LOAD IMAGES
# -----------------------------------
img1 = Image.open(
    r"C:\Users\raghu\Downloads\SRN1326-24Recreated.png"
)

img2 = Image.open(
    r"C:\Users\raghu\Downloads\SRN1326-24GEN.jpg"
)


# -----------------------------------
# REMOVE TRANSPARENCY
# -----------------------------------
img1 = remove_transparency(img1)
img2 = remove_transparency(img2)


# -----------------------------------
# RESIZE FOR DISPLAY
# -----------------------------------
img1_resized = img1.resize((224, 224))
img2_resized = img2.resize((224, 224))


# -----------------------------------
# TRANSFORM IMAGES
# -----------------------------------
img1_tensor = transform(img1).unsqueeze(0).to(device)
img2_tensor = transform(img2).unsqueeze(0).to(device)


# -----------------------------------
# GPU WARMUP
# -----------------------------------
with torch.no_grad():
    _ = vector_model(img1_tensor)


# -----------------------------------
# START TIMER
# -----------------------------------
start_time = time.time()


# -----------------------------------
# FEATURE EXTRACTION
# -----------------------------------
with torch.no_grad():

    # -------------------------------
    # 1D FEATURES
    # -------------------------------
    feat1 = vector_model(img1_tensor)
    feat2 = vector_model(img2_tensor)

    feat1 = feat1.view(feat1.size(0), -1)
    feat2 = feat2.view(feat2.size(0), -1)

    vector_similarity = F.cosine_similarity(
        feat1,
        feat2
    )

    # -------------------------------
    # MATRIX FEATURES
    # -------------------------------
    mat1 = matrix_model(img1_tensor)
    mat2 = matrix_model(img2_tensor)

    mat1_flat = mat1.view(mat1.size(0), -1)
    mat2_flat = mat2.view(mat2.size(0), -1)

    matrix_similarity = F.cosine_similarity(
        mat1_flat,
        mat2_flat
    )


# -----------------------------------
# GPU SYNC FOR ACCURATE TIMING
# -----------------------------------
if torch.cuda.is_available():
    torch.cuda.synchronize()


# -----------------------------------
# END TIMER
# -----------------------------------
end_time = time.time()

inference_time_ms = (
    end_time - start_time
) * 1000


# -----------------------------------
# SCORES
# -----------------------------------
vector_score = vector_similarity.item()
matrix_score = matrix_similarity.item()

final_score = (
    0.7 * vector_score +
    0.3 * matrix_score
)


# -----------------------------------
# SHOW IMAGES + SCORES
# -----------------------------------
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6)
)

axes[0].imshow(img1_resized)
axes[0].set_title(
    f"Image 1\n"
    f"Vector: {vector_score:.4f}\n"
    f"Matrix: {matrix_score:.4f}\n"
    f"Final : {final_score:.4f}"
)
axes[0].axis("off")

axes[1].imshow(img2_resized)
axes[1].set_title(
    f"Image 2\n"
    f"Vector: {vector_score:.4f}\n"
    f"Matrix: {matrix_score:.4f}\n"
    f"Final : {final_score:.4f}"
)
axes[1].axis("off")

plt.tight_layout()
plt.show()


# -----------------------------------
# PRINT RESULTS
# -----------------------------------
print("\n----- RESULTS -----")

print("1D Feature Shape :", feat1.shape)
print("Matrix Shape     :", mat1.shape)

print(f"\n1D Vector Similarity : {vector_score:.4f}")
print(f"Matrix Similarity    : {matrix_score:.4f}")
print(f"Final Hybrid Score   : {final_score:.4f}")

print(f"\nInference Time       : {inference_time_ms:.2f} ms")