import os
import shutil
import random

SOURCE = "dataset"
DESTINATION = "split_dataset"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

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

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# Create destination folders
for split in ["train", "validation", "test"]:
    for class_name in classes:
        os.makedirs(
            os.path.join(DESTINATION, split, class_name),
            exist_ok=True
        )

print("\n========== DATASET SPLITTING ==========\n")

total_train = 0
total_validation = 0
total_test = 0

for class_name in classes:

    source_folder = os.path.join(SOURCE, class_name)

    images = [
        f for f in os.listdir(source_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    for filename in train_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(
                DESTINATION,
                "train",
                class_name,
                filename
            )
        )

    for filename in validation_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(
                DESTINATION,
                "validation",
                class_name,
                filename
            )
        )

    for filename in test_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(
                DESTINATION,
                "test",
                class_name,
                filename
            )
        )

    total_train += len(train_images)
    total_validation += len(validation_images)
    total_test += len(test_images)

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(validation_images)}, "
        f"Test={len(test_images)}"
    )

print("\n---------------------------------------")
print("TOTAL TRAINING:", total_train)
print("TOTAL VALIDATION:", total_validation)
print("TOTAL TEST:", total_test)
print("TOTAL:", total_train + total_validation + total_test)
print("---------------------------------------")