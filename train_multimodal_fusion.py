import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_FILE = "real_dataset/real_multimodal_data.csv"

if not os.path.exists(CSV_FILE):
    print(f"[!] Dataset file '{CSV_FILE}' not found. Run collect_real_dataset.py first.")
    exit()

df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} physical multimodal samples.")

sensor_features = ["temp_c", "humidity_pct", "gas_voltage", "color_r", "color_g", "color_b"]
X_sensor = df[sensor_features].values
y = df["label"].values

scaler = StandardScaler()
X_sensor_scaled = scaler.fit_transform(X_sensor)

def load_real_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img.numpy()

X_images = np.array([load_real_image(p) for p in df["image_path"]])

X_img_train, X_img_val, X_sen_train, X_sen_val, y_train, y_val = train_test_split(
    X_images, X_sensor_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Vision Branch
base_vision = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
base_vision.trainable = False

image_input = layers.Input(shape=(224, 224, 3), name="image_input")
v = base_vision(image_input)
v = layers.GlobalAveragePooling2D()(v)
v = layers.Dense(128, activation="relu")(v)

# Sensor MLP Branch
sensor_input = layers.Input(shape=(6,), name="sensor_input")
s = layers.Dense(64, activation="relu")(sensor_input)
s = layers.Dense(32, activation="relu")(s)

# Fusion Layer
fused = layers.concatenate([v, s])
fused = layers.Dense(64, activation="relu")(fused)
fused = layers.Dropout(0.3)(fused)
output = layers.Dense(8, activation="softmax", name="output")(fused)

multimodal_model = models.Model(inputs=[image_input, sensor_input], outputs=output)
multimodal_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

multimodal_model.fit(
    {"image_input": X_img_train, "sensor_input": X_sen_train},
    y_train,
    validation_data=({"image_input": X_img_val, "sensor_input": X_sen_val}, y_val),
    epochs=15,
    batch_size=16
)

os.makedirs("models", exist_ok=True)
multimodal_model.save("models/multimodal_fusion_model.keras")
print("\n[SUCCESS] Multimodal fusion model saved to 'models/multimodal_fusion_model.keras'")