import os
import tensorflow as tf

MODEL_PATH = os.path.join("models", "vegetable_classifier_mobilenetv2.keras")
TFLITE_PATH = os.path.join("models", "vegetable_classifier.tflite")

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Convert to TFLite format
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Standard edge optimization

tflite_model = converter.convert()

# Save the TFLite model
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

keras_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
tflite_size = os.path.getsize(TFLITE_PATH) / (1024 * 1024)

print(f"\n--- Conversion Complete ---")
print(f"Original Keras Model Size: {keras_size:.2f} MB")
print(f"Optimized TFLite Model Size: {tflite_size:.2f} MB")
print(f"Saved to: {TFLITE_PATH}")