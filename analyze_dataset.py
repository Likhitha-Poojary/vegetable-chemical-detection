import os
import random
import matplotlib.pyplot as plt
from PIL import Image

BASE = "dataset"

classes = [
    "FreshApple",
    "RottenApple",
    "FreshBanana",
    "RottenBanana",
    "FreshPotato",
    "RottenPotato",
    "FreshTomato",
    "RottenTomato"
]

fig, axes = plt.subplots(8, 5, figsize=(15, 24))

for row, class_name in enumerate(classes):

    folder = os.path.join(BASE, class_name)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

    samples = random.sample(files, min(5, len(files)))

    for col in range(5):

        ax = axes[row, col]

        if col < len(samples):

            filepath = os.path.join(folder, samples[col])

            image = Image.open(filepath).convert("RGB")

            ax.imshow(image)
            ax.set_title(class_name)

        ax.axis("off")

plt.tight_layout()

output = "dataset_sample_grid.png"
plt.savefig(output, dpi=150)

print("Sample grid saved as:", output)

plt.show()