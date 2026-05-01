import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
import torch

# Load image
image = Image.open("image.jpg").convert("RGB")

# Load SAM from Hugging Face
print("Loading SAM model...")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base")
model.eval()

# Run automatic segmentation using a grid of points as prompts
width, height = image.size
grid_points = []
for x in range(1, 5):
    for y in range(1, 5):
        grid_points.append([int(width * x / 5), int(height * y / 5)])

inputs = processor(
    images=image,
    input_points=[[grid_points]],
    return_tensors="pt"
)

print("Running segmentation...")
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks,
    inputs["original_sizes"],
    inputs["reshaped_input_sizes"]
)[0][0]

scores = outputs.iou_scores[0][0]

# Pick top 6 masks by confidence score
top_indices = scores.argsort(descending=True)[:6]

# Display results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Segmented image
axes[1].imshow(image)
colors = ["red", "green", "blue", "yellow", "magenta", "cyan"]
patches = []
for i, idx in enumerate(top_indices):
    mask = masks[idx].numpy()
    color = plt.cm.colors.to_rgba(colors[i], alpha=0.4)
    colored_mask = np.zeros((*mask.shape, 4))
    colored_mask[mask] = color
    axes[1].imshow(colored_mask)
    patches.append(mpatches.Patch(color=colors[i], alpha=0.6, label=f"Segment {i+1} (score: {scores[idx]:.2f})"))

axes[1].legend(handles=patches, loc="upper right", fontsize=8)
axes[1].set_title("SAM Segmentation (top 6 masks)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("result.png", dpi=150)
plt.show()
print("Done. Result saved as result.png")