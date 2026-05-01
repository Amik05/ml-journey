import matplotlib
matplotlib.use('MacOSX')
import numpy as np
import json
from PIL import Image
import torch
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
from transformers import pipeline as hf_pipeline
# import open3d as o3d
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── 1. Load image and points ───────────────────────────────────────────────────
image = Image.open("image.jpg").convert("RGB")
image_np = np.array(image)

with open("points.json", "r") as f:
    data = json.load(f)

positive_points = data["positive"]
negative_points = data["negative"]
all_points = positive_points + negative_points
all_labels = [1] * len(positive_points) + [0] * len(negative_points)

print(f"Loaded {len(positive_points)} positive and {len(negative_points)} negative points.")

# ── 2. Run SAM ─────────────────────────────────────────────────────────────────
print("Loading SAM...")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base")
model.eval()

print("Running SAM segmentation...")
inputs = processor(
    images=image,
    input_points=[[all_points]],
    input_labels=[[all_labels]],
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks,
    inputs["original_sizes"],
    inputs["reshaped_input_sizes"]
)[0][0]

scores = outputs.iou_scores[0][0]
best_mask = masks[scores.argmax()].numpy()
print(f"Segmentation done. Best mask score: {scores.max():.2f}")

# ── 3. Estimate depth ──────────────────────────────────────────────────────────
print("Loading Depth Anything...")
depth_estimator = hf_pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)

print("Estimating depth...")
depth_result = depth_estimator(image)
depth_map = np.array(depth_result["depth"])
depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

# ── 4. Save intermediate results ───────────────────────────────────────────────
print("Saving results...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image)
for p in positive_points:
    axes[0].plot(p[0], p[1], 'g*', markersize=15)
for p in negative_points:
    axes[0].plot(p[0], p[1], 'r*', markersize=15)
axes[0].set_title("Prompts")
axes[0].axis("off")

axes[1].imshow(image)
colored_mask = np.zeros((*best_mask.shape, 4))
colored_mask[best_mask] = [0, 1, 0, 0.5]
axes[1].imshow(colored_mask)
axes[1].set_title(f"SAM Mask (score: {scores.max():.2f})")
axes[1].axis("off")

axes[2].imshow(depth_normalized, cmap="plasma")
axes[2].set_title("Estimated Depth")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("result_all.png", dpi=150)
plt.show()

# ── 5. Build 3D point cloud ────────────────────────────────────────────────────
print("Building 3D point cloud...")

depth_resized = np.array(
    Image.fromarray(depth_normalized).resize(
        (image_np.shape[1], image_np.shape[0]),
        Image.BILINEAR
    )
)

height, width = image_np.shape[:2]
points_3d = []
colors_3d = []

for y in range(0, height, 2):  # step by 2 to go faster
    for x in range(0, width, 2):
        if best_mask[y, x]:
            z = depth_resized[y, x]
            x_3d = (x - width / 2) / width
            y_3d = -(y - height / 2) / height
            z_3d = z
            points_3d.append([x_3d, y_3d, z_3d])
            colors_3d.append(image_np[y, x] / 255.0)

points_3d = np.array(points_3d)
colors_3d = np.array(colors_3d)
print(f"Point cloud has {len(points_3d)} points.")

# ── 6. Visualize with matplotlib ───────────────────────────────────────────────
print("Rendering 3D point cloud...")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Subsample for speed
step = max(1, len(points_3d) // 5000)
pts = points_3d[::step]
cls = colors_3d[::step]

ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
           c=cls, s=0.5, depthshade=True)

ax.set_title("3D Point Cloud")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Depth")

plt.tight_layout()
plt.savefig("result_3d.png", dpi=150)
plt.show()

print("Done. Saved as result_3d.png")