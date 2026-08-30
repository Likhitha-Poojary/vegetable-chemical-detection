import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Configuration
MODEL_PATH = os.path.join("models", "vegetable_classifier_mobilenetv2.keras")
CLASS_NAMES = [
    'FreshApple', 'FreshBanana', 'FreshPotato', 'FreshTomato', 
    'RottenApple', 'RottenBanana', 'RottenPotato', 'RottenTomato'
]

# 2. Load the trained model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.\n")

def classify_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: Could not find '{img_path}'")
        return

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch of 1

    # Predict
    predictions = model.predict(img_array, verbose=0)
    
    # Calculate confidence
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = 100 * predictions[0][predicted_class_index]

    print("-" * 30)
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    while True:
        img_path = input("Enter image path (or type 'quit' to exit): ").strip('"\'')
        if img_path.lower() == 'quit':
            break
        classify_image(img_path)