import cv2
import numpy as np
from keras.preprocessing import image
from CarTypeClassification import CarTypeClassification

class ObjectDetection:

    def init_yolo_model(self):
        """
        The method reads the weights and does initializes yolo object detection model
        :return: None
        """
        self.net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
        with open('coco.names','r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))


    def process_image(self, img):
        """
        TODO
        :return:
        """
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        return img_tensor

    def detect_car(self, blob, height, width, frame):
        """
        The method initializes the yolo model, detects the objects in the current blob and
        :parameter blob: Section of frame in size 416x416 for object detection
        :parameter height: height of the frame for ___
        :parameter width: width of the frame for ___
        :parameter frame: ___
        :return:
        """
        self.init_yolo_model()
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        class_ids = []
        confidences = []
        boxes = []
        car_predictions = []

        # Check for detections in output
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Detecting bounding boxes when object is detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])

                # Cropping the bounding boxes if car is detected
                if label == 'Car' or label == 'car':
                    print("Car detected")
                    car = frame[y:y + h, x:x + w]
                    car_tensor = self.process_image(car)
                    print("here, ",car_tensor)
                    car_predictions.append(car_tensor)

        # Calling the next module in pipeline to type of the car(sedan or hatchback)
        print("Cars in the frame:", len(car_predictions))
        CarTypeClassification().detect_car_type(car_predictions)