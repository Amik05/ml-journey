import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image

image = Image.open("image.jpg").convert("RGB")

print("Left click = include | Right click = exclude | Close when done.")
fig, ax = plt.subplots()
ax.imshow(image)
ax.set_title("Left click = include | Right click = exclude | Close when done")

positive_points, negative_points = [], []

def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    if event.button == 1:
        positive_points.append([x, y])
        ax.plot(x, y, 'g*', markersize=15)
        print(f"Positive: ({x}, {y})")
    elif event.button == 3:
        negative_points.append([x, y])
        ax.plot(x, y, 'r*', markersize=15)
        print(f"Negative: ({x}, {y})")
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Save points to file
data = {"positive": positive_points, "negative": negative_points}
with open("points.json", "w") as f:
    json.dump(data, f)

print(f"Saved {len(positive_points)} positive and {len(negative_points)} negative points.")