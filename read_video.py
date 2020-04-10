from keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
from enum import Enum
import cv2
import numpy as np
import threading
import queue
import time

frameQueue = queue.Queue()
carQueue = queue.Queue()
colourQueue = queue.Queue()
outputQueue = queue.Queue()


video = "video.mp4"

class Colour(Enum):
    BLACK = 1
    SILVER = 2
    RED = 3
    WHITE = 4
    BLUE = 5


# Class to read video, put each frame into queue for pipeline to read
class VideoReader(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(VideoReader,self).__init__()
        self.target = target
        self.name = name
        self.video = cv2.VideoCapture(video)
        self.fps = 0.03 # frame every 0.03 seconds
    
    def run(self):
        frameNo = 1
        while self.video.isOpened():
            start = time.time()
            ret, frame = self.video.read()
            if ret:
                frameQueue.put((frame, frameNo))
                frameNo += 1
                diff = time.time() - start
                # Wait until 0.03 seconds passed for next frame
                while diff < self.fps:
                    diff = time.time() - start
            else:
                break
                
# Class for detecting cars
class ObjectDetector(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ObjectDetector,self).__init__()
        self.target = target
        self.name = name
        
        self.yolo = cv2.dnn.readNet('yolov3-tiny.weights','yolov3-tiny.cfg')
        self.layer_names = self.yolo.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.yolo.getUnconnectedOutLayers()]
        self.classes = self.readClassNames('coco.names')
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
    
    def readClassNames(self, classFile):
        with open(classFile,'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    
    def run(self):
        while True:
            if not frameQueue.empty():
                frame, frameNo = frameQueue.get()
                cars, flag = self.processFrame(frame)
                if flag:  
                    # Car detected, push through pipeline
                    carQueue.put((frame, cars, frameNo))
                else:
                    print("Frame Number", frameNo)
                    print("No car this frame")
                    # No car, save frame as is
                    #f = f'frame{frameNo}.png'
                    #cv2.imwrite(f, frame)
                
                
    def processFrame(self, frame):
        height, width, channels = frame.shape
        # Resize for YOLO
        resized = cv2.resize(frame, None, fx=0.4, fy=0.4)
        blob = cv2.dnn.blobFromImage(resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        # Set input, forward pass through YOLO
        self.yolo.setInput(blob)
        outputs = self.yolo.forward(self.output_layers)
        class_ids = []
        confidences = []
        boxes = []
        # Loop through outputs for objects
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        cars = []
        flag = False
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                # Crop image if car
                if label == 'Car' or label == 'car':
                    
                    flag = True
                    car = frame[y:y+h, x:x+w]
                    
                    cars.append((car, boxes[i]))
                    
        return cars, flag
                    
                        
class TypeClassifier(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(TypeClassifier,self).__init__()
        self.target = target
        self.name = name
        
        self.model = self.loadModel('model.json', 'model.h5')
        
    def loadModel(self, model, weights):
        # Load model from json file and weights file
        json = open(model, 'r')
        loaded_model = json.read()
        json.close()
        model = model_from_json(loaded_model)
        model.load_weights(weights)
        return model
    
    # Convert image to tensor for MobileNet
    def process_image(self, img):
        try:    
            img = cv2.resize(img, (224, 224))
        except Exception as e:
            pass
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        return img_tensor
    
    def run(self):
        while True:
            if not carQueue.empty():
                frame, cars, frameNo = carQueue.get()
                predictions = self.predictCar(cars)
                colourQueue.put((frame, cars, predictions, frameNo))
                
    
    def predictCar(self, cars):
        predictions = []
        for car, box in cars:
            car_tensor = self.process_image(car)
            prediction = self.model.predict(car_tensor)
            predictions.append(np.argmax(prediction[0]))
        return predictions
                
class ColourClassifier(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ColourClassifier,self).__init__()
        self.target = target
        self.name = name
        
        self.colourRanges = self.makeRanges()
    
    def makeRanges(self):
        blackRange = [np.array([0,0,0]), np.array([255, 255, 40])]
        silverRange = [np.array([0,0,40]), np.array([255, 10, 180])]
        redRange = [np.array([20, 140, 30]), np.array([255, 255, 200])]
        whiteRange = [np.array([0, 0, 230]), np.array([180, 25, 255])]
        blueRange = [np.array([100,50,50]), np.array([130,255,255])]
        return [blackRange, silverRange, redRange, whiteRange, blueRange]
        
    def run(self):
        while True:
            if not colourQueue.empty():
                frame, cars, predictions, frameNo = colourQueue.get()
                colours = self.classify_colour(cars)
                outputQueue.put((frame, cars, predictions, colours, frameNo))
                
                
    
    def classify_colour(self, cars):
        # Classify colour by calculating mask for each colour range
        # Mask has 0 for pixels outside range, 255 for pixels in range
        # Count number of 255s, highest number corresponds to colour of car
        carColours = []
        for car, box in cars:
            
            pixel_counts = []
            carHSV = cv2.cvtColor(car, cv2.COLOR_BGR2HSV)
            for colourRange in self.colourRanges:
                mask = cv2.inRange(carHSV, colourRange[0], colourRange[1])
                pixel_count = np.count_nonzero(mask == 255)
                pixel_counts.append(pixel_count)
               
            carColours.append(np.argmax(pixel_counts) + 1)
        # Index corresponds to colour
        return carColours
        
                
class Output(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(Output,self).__init__()
        self.target = target
        self.name = name
                
                
    def run(self):
        while True:
            if not outputQueue.empty():
                frame, cars, predictions, colours, frameNo = outputQueue.get()
                print("Frame Number", frameNo)
                print("Predictions", predictions)
                print("Colours", colours)

if __name__ == "__main__":
    VR = VideoReader(name="VideoReader")
    OD = ObjectDetector(name="ObjectDetector")
    TC = TypeClassifier(name="TypeClassifier")
    CC = ColourClassifier(name="ColourClassifier")
    O = Output(name="Output")
    
    print("Starting threads")
    VR.start()
    OD.start()
    TC.start()
    CC.start()
    O.start()
    print("All threads started")
                
