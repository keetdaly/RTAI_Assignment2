import cv2
import numpy as np

class ObjectDetection:

    def yolo_setup(self):
        """
        The method reads the weights and does setup for yolo model
        :return:
        """
        # Set up Yolo
        self.net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
        with open('coco.names','r') as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(classes), 3))

    def detect_car(self, blob, height, width):
        self.yolo_setup()
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        print("Forward pass through YOLO")
        class_ids = []
        confidences = []
        boxes = []
        # Check for detections in output
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    print("Object detected")
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
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
                label = str(classes[class_ids[i]])
                # Crop image if car
                if label == 'Car' or label == 'car':
                    print("Car detected")
                    car = frame[y:y + h, x:x + w]
                    car_tensor = process_image(car)
                    prediction = model.predict(car_tensor)
                    frame_predictions.append(prediction[0].tolist())
                else:
                    frame_predictions.append([0, 0])