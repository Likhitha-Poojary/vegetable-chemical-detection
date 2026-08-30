import os
from PIL import Image
import imagehash
from collections import Counter

BASE = "dataset"

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

hashes = {}
duplicates = []

for class_name in sorted(os.listdir(BASE)):

    folder = os.path.join(BASE, class_name)

    if not os.path.isdir(folder):
        continue

    for filename in os.listdir(folder):

        if os.path.splitext(filename)[1].lower() not in extensions:
            continue

        filepath = os.path.join(folder, filename)

        try:
            with Image.open(filepath) as img:
                h = str(imagehash.phash(img))

            if h in hashes:
                original_file, original_class = hashes[h]

                duplicates.append(
                    (filepath, class_name, original_file, original_class)
                )
            else:
                hashes[h] = (filepath, class_name)

        except Exception as e:
            print("Error:", filepath, e)


print("\n========== DUPLICATE ANALYSIS ==========\n")

print("Unique image hashes:", len(hashes))
print("Possible duplicate pairs:", len(duplicates))

same_class = 0
different_class = 0

for new_file, new_class, original_file, original_class in duplicates:

    if new_class == original_class:
        same_class += 1
    else:
        different_class += 1

print("\nSame-class possible duplicates:", same_class)
print("Different-class possible duplicates:", different_class)

print("\n--- DIFFERENT-CLASS POSSIBLE DUPLICATES ---")

for new_file, new_class, original_file, original_class in duplicates:

    if new_class != original_class:

        print("\nNEW:")
        print(new_class, "->", new_file)

        print("ORIGINAL:")
        print(original_class, "->", original_file)

print("\n========================================")