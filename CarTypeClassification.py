prediction = self.model.predict(car_tensor)
                    frame_predictions.append(prediction[0].tolist())
                else:
                    frame_predictions.append([0, 0])