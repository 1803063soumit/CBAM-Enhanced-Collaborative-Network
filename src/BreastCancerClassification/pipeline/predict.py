import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
    def predict(self):
        # load model
        model = load_model(os.path.join("models", "trained_model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.
        predictions = model.predict(test_image)
        result = (predictions > 0.5).astype(int)
        print("--------Prediction------->", predictions)

        if result[0] == 1:
            prediction = 'Malignant'
            return [{"image": prediction}]
        else:
            prediction = 'Benign'
            return [{"image": prediction}]

        # # Evaluate the model on the test set
        # predictions = model.predict(test_generator)
        #
        # predicted_classes = (predictions > 0.5).astype(int)
        # true_classes = test_generator.classes