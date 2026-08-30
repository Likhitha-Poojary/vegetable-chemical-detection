import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

TEST_DIR = "split_dataset/test"
MODEL_PATH = "models/vegetable_classifier_mobilenetv2.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# 1. Load Model and Test Data
model = tf.keras.models.load_model(MODEL_PATH)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

# 2. Predict on Test Set
y_true = []
y_pred_probs = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred_probs.extend(preds)

y_true = np.array(y_true)
y_pred = np.argmax(y_pred_probs, axis=1)

# 3. Print Classification Metrics
print("\n" + "="*55)
print("             EVALUATION REPORT")
print("="*55)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# 4. Save and Show Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - MobileNetV2 Vegetable Classifier")
plt.ylabel("Ground Truth")
plt.xlabel("Predicted Class")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

os.makedirs("results", exist_ok=True)
plot_path = "results/confusion_matrix.png"
plt.savefig(plot_path, dpi=300)
print(f"\n[Saved] Confusion matrix plot exported to: {plot_path}")
plt.show()