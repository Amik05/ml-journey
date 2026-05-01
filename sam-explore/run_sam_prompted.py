import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
import torch

# Load image
image = Image.open("image.jpg").convert("RGB")

# Show image and let user click multiple points
print("Left click = positive point (include)")
print("Right click = negative point (exclude)")
print("Close the window when done.")

fig, ax = plt.subplots()
ax.imshow(image)
ax.set_title("Left click = include | Right click = exclude | Close when done")

positive_points = []
negative_points = []

def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    if event.button == 1:  # left click
        positive_points.append([x, y])
        ax.plot(x, y, 'g*', markersize=15)
        print(f"Positive point: ({x}, {y})")
    elif event.button == 3:  # right click
        negative_points.append([x, y])
        ax.plot(x, y, 'r*', markersize=15)
        print(f"Negative point: ({x}, {y})")
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

if not positive_points:
    print("No points selected, exiting.")
    exit()

# Build point labels: 1 = positive, 0 = negative
all_points = positive_points + negative_points
all_labels = [1] * len(positive_points) + [0] * len(negative_points)

# Load SAM
print("Loading SAM model...")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base")
model.eval()

# Run segmentation
print("Running segmentation...")
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

# Display result
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(image)
for p in positive_points:
    axes[0].plot(p[0], p[1], 'g*', markersize=15, label="include")
for p in negative_points:
    axes[0].plot(p[0], p[1], 'r*', markersize=15, label="exclude")
axes[0].set_title("Your prompts (green=include, red=exclude)")
axes[0].axis("off")

axes[1].imshow(image)
colored_mask = np.zeros((*best_mask.shape, 4))
colored_mask[best_mask] = [0, 1, 0, 0.5]
axes[1].imshow(colored_mask)
axes[1].set_title(f"Segmented object (score: {scores.max():.2f})")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("result_prompted.png", dpi=150)
plt.show()
print("Done. Result saved as result_prompted.png")