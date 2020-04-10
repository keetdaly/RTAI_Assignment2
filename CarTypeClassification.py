from keras.models import model_from_json
from ColourAttributeClassifier import ColourAttributeClassifier

class CarTypeClassification:

    def init_mobile_net_model(self):
        """
        The method reads the car type detection model
        :return:
        """
        # Set up MobileNet model
        json_file = open('model.json', 'r')
        loaded_model = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model)
        self.model.load_weights('model.h5')

    def detect_car_type(self, car_predictions):
        self.init_mobile_net_model()
        cartype_predictions = []

        for car_tensor in car_predictions:
            prediction = self.model.predict(car_tensor)
            if (prediction[0][0] > 0.9):
                print("Hatchback found", prediction[0][0])
                # cartype_predictions.append("Hatchback")
            elif (prediction[0][1] > 0.9):
                print("Sedan found", prediction[0][1])
                #cartype_predictions.append("Sedan")
            else:
                print("Cannot classify cartype")
                #cartype_predictions.append("Unknown")

        print(cartype_predictions)
        ColourAttributeClassifier().detect_car_colour(car_predictions)
