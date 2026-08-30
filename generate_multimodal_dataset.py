import os
import random
import pandas as pd
import numpy as np

BASE_DIR = "split_dataset"
CSV_DIR = "multimodal_records"
os.makedirs(CSV_DIR, exist_ok=True)

CLASS_MAP = {
    'FreshApple': 0, 'FreshBanana': 1, 'FreshPotato': 2, 'FreshTomato': 3,
    'RottenApple': 4, 'RottenBanana': 5, 'RottenPotato': 6, 'RottenTomato': 7
}

def generate_sensor_profile(class_name):
    is_rotten = "Rotten" in class_name
    if is_rotten:
        temp = np.random.normal(30.5, 1.5)
        hum = np.random.normal(78.0, 4.0)
        gas = np.random.normal(420.0, 50.0)
        r = np.random.normal(110, 15)
        g = np.random.normal(85, 12)
        b = np.random.normal(60, 10)
    else:
        temp = np.random.normal(23.0, 1.2)
        hum = np.random.normal(55.0, 3.5)
        gas = np.random.normal(65.0, 15.0)
        r = np.random.normal(190, 20)
        g = np.random.normal(170, 18)
        b = np.random.normal(120, 15)
    return [temp, hum, gas, r, g, b]

for split in ["train", "validation", "test"]:
    records = []
    split_path = os.path.join(BASE_DIR, split)
    
    for class_name in os.listdir(split_path):
        class_folder = os.path.join(split_path, class_name)
        if not os.path.isdir(class_folder):
            continue
        
        for fname in os.listdir(class_folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_img_path = os.path.join(class_folder, fname)
                sensors = generate_sensor_profile(class_name)
                records.append([
                    full_img_path,
                    CLASS_MAP[class_name],
                    *sensors
                ])
    
    df = pd.DataFrame(records, columns=[
        "image_path", "label", "temp", "hum", "gas", "color_r", "color_g", "color_b"
    ])
    out_file = os.path.join(CSV_DIR, f"{split}_multimodal.csv")
    df.to_csv(out_file, index=False)
    print(f"Generated {out_file} with {len(df)} samples.")