import os
import glob
import time
import cv2
import numpy as np
import tensorflow as tf

TFLITE_PATH = "models/vegetable_classifier.tflite"
TEST_DIR = "split_dataset/test"

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_paths = glob.glob(os.path.join(TEST_DIR, "*", "*.jpg")) + glob.glob(os.path.join(TEST_DIR, "*", "*.png"))
sample_paths = image_paths[:100]

inference_times = []
total_times = []

print(f"Benchmarking inference latency on {len(sample_paths)} real test set images...")

for path in sample_paths:
    t0 = time.perf_counter()
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img / 127.5) - 1.0
    input_data = np.expand_dims(img, axis=0).astype(np.float32)
    t1 = time.perf_counter()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
    t2 = time.perf_counter()

    inference_times.append((t2 - t1) * 1000)
    total_times.append((t2 - t0) * 1000)

avg_inf = np.mean(inference_times)
avg_total = np.mean(total_times)
fps = 1000.0 / avg_total
file_size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)

print("\n" + "="*55)
print("     REAL-IMAGE TFLITE EDGE BENCHMARK RESULTS")
print("="*55)
print(f"Model File: {TFLITE_PATH}")
print(f"Model File Size: {file_size_mb:.2f} MB")
print(f"Pure NN Inference Latency: {avg_inf:.2f} ms")
print(f"End-to-End Latency (Disk + Preprocess + NN): {avg_total:.2f} ms")
print(f"Real Pipeline Throughput: {fps:.1f} FPS")
print("="*55)