import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import sys

def classify_image(image_path):
    # Load the pre-trained model
    model = MobileNetV2(weights='imagenet')

    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Make a prediction
    predictions = model.predict(image)
    predictions = decode_predictions(predictions, top=5)[0]

    # Print the predictions
    for (imagenetID, label, prob) in predictions:
        print(f"{label}: {prob*100:.2f}%")

if __name__ == "__main__":
    classify_image(sys.argv[1])
