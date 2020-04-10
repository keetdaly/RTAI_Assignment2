import cv2
import numpy as np

class ColourAttributeClassifier:

    def detect_car_colour(self, cartype_predictions):
        for car in cartype_predictions:
            data = np.reshape(car, (-1, 3))
            print(car.tostring())
            data = np.float32(data)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            flags = cv2.KMEANS_RANDOM_CENTERS
            compactness, labels, centers = cv2.kmeans(data, 1, None, criteria, 10, flags)

            print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))