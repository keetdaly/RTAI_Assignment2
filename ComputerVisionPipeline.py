from keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from enum import Enum
import cv2
import numpy as np
import threading
import queue
import time
import csv

### Initializing shared resources for the classes ###

# Initializing pipeline queues
frameQueue = queue.Queue()
carQueue = queue.Queue()
colourQueue = queue.Queue()
outputQueue = queue.Queue()

# video file path
video = "video.mp4"

# List for recording colour of each car type
# index: 0 -> Black, 1 -> Silver, 2 -> Red, 3 -> White, 4 -> Blue
HATCHBACK = [0,0,0,0,0]
SEDAN = [0,0,0,0,0]

# Enum for colour detection
class Colour(Enum):
    BLACK = 1
    SILVER = 2
    RED = 3
    WHITE = 4
    BLUE = 5

### Class to read video and add the frames into pipeline queue to process ###
class VideoReader(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(VideoReader,self).__init__()
        self.target = target
        self.name = name
        self.video = cv2.VideoCapture(video)
        self.fps = 0.03 # reading frame every 0.03 seconds
    
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
                # End of video, lets cascade know so loops can be exited
                frameQueue.put(([], 0))
                break
                
### Class for detecting objects and adding the cars to the queue for further processing ###
class ObjectDetector(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(ObjectDetector,self).__init__()
        self.target = target
        self.name = name
        
        # Setting up TinyYOLO with weights and config
        self.yolo = cv2.dnn.readNet('yolov3-tiny.weights','yolov3-tiny.cfg')
        self.layer_names = self.yolo.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.yolo.getUnconnectedOutLayers()]
        self.classes = self.readClassNames('coco.names')        
    
    def readClassNames(self, classFile):
        with open(classFile,'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    
    def run(self):
        event_extraction_times = []
        while True:
            if not frameQueue.empty():
                frame, frameNo = frameQueue.get()
                if frameNo == 0:
                    # No more frames to process
                    # Tell cascade no more frames to process
                    # Break from loop to write to file
                    carQueue.put((frame, [], frameNo))
                    break

                # Time in milliseconds
                start_milli = int(time.time() * 1000)
                cars = self.processFrame(frame)
                extraction_time_milli = int(time.time() * 1000) - start_milli
                event_extraction_times.append(extraction_time_milli)
                
                for car, box in cars:
                    # Some bounding boxes may have invalid dimensions and cause
                    # MobileNet to crash
                    if car.shape[0] <= 0 or car.shape[1] <= 0:
                        #print("Invalid dimensions... removing car from list")
                        cars.remove((car, box))

                if cars:
                    # Car detected, push through pipeline
                    carQueue.put((frame, cars, frameNo))
                else:
                    # No car
                    # Let cascade know that no car detected -> set extraction time to 0 
                    # This allows timestamps in lists to be aligned by frame for processing
                    carQueue.put((frame, cars, frameNo))
                    # Save frame as is
                    f = f'results/frame{frameNo}.png'
                    cv2.imwrite(f, frame)
                    data = [frameNo] + [0] * 11

                    with open("predictions.csv","a") as f:
                        w = csv.writer(f)
                        w.writerow(data)

        # No more frames to process, write extraction times to file for processing
        with open("Q1_extraction_times.csv", "a") as f:
            w = csv.writer(f)
            w.writerow(event_extraction_times)
                
    def processFrame(self, frame):
        h, w, channels = frame.shape

        # Resizing image for YOLO
        resized = cv2.resize(frame, None, fx=0.4, fy=0.4)
        blob = cv2.dnn.blobFromImage(resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        # Setting up input, forward pass through YOLO
        self.yolo.setInput(blob)
        outputs = self.yolo.forward(self.output_layers)
        class_ids = []
        confidences = []
        boxes = []

        # Looping through outputs for objects detected
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:

                    # Calculating coordinates for bounding box
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Calculating the bounding boxes for objects
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
        cars = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])

                # Cropping the image if car
                if label == 'Car' or label == 'car':
                    car = frame[y:y+h, x:x+w]
                    cars.append((car, boxes[i]))
        return cars

### Class to detect the car type ###
class TypeClassifier(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(TypeClassifier,self).__init__()
        self.target = target
        self.name = name
        
        self.model = self.loadModel('model.json', 'model.h5')
        
    def loadModel(self, model, weights):
        # Loading model from json file and weights file
        json = open(model, 'r')
        loaded_model = json.read()
        json.close()
        model = model_from_json(loaded_model)
        model.load_weights(weights)
        return model
    
    def process_image(self, img):
        # Converting image to tensor for MobileNet
        img = cv2.resize(img, (224,224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        return img_tensor
    
    def run(self):
        event_extraction_times = []
        while True:
            if not carQueue.empty():
                frame, cars, frameNo = carQueue.get()

                if frameNo == 0:
                    # No more frames
                    # Tell next classifier no more frames
                    # Break from loop
                    colourQueue.put((frame, cars, [], frameNo))
                    break

                if not cars:
                    # No car in frame, no need to process
                    # Set extraction time to 0, tell next classifier no car detected
                    event_extraction_times.append(0.0)
                    colourQueue.put((frame, cars, [], frameNo))
                    continue

                start_time_milli = int(time.time() * 1000)
                predictions = self.predictCar(cars)
                extraction_time_milli = int(time.time() * 1000) - start_time_milli
                event_extraction_times.append(extraction_time_milli)
                colourQueue.put((frame, cars, predictions, frameNo))

        # Writing timestamps to file for processing
        with open("Q2_extraction_times.csv", "a") as f:
            w = csv.writer(f)
            w.writerow(event_extraction_times)
    
    def predictCar(self, cars):
        # Using MobileNet to predict Hatchback or Sedan for cars detected in each frame
        predictions = []
        for car, box in cars:
            car_tensor = self.process_image(car)
            prediction = self.model.predict(car_tensor)
            predictions.append(np.argmax(prediction[0]))
        return predictions

### Class to detect the colour of the car from the given 5 categories ###
class ColourClassifier(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(ColourClassifier,self).__init__()
        self.target = target
        self.name = name
        
        self.colourRanges = self.makeRanges()
    
    def makeRanges(self):
        # HSV ranges for colours
        blackRange = [np.array([0,0,0]), np.array([255, 255, 40])]
        silverRange = [np.array([0,0,40]), np.array([255, 10, 180])]
        redRange = [np.array([20, 140, 30]), np.array([255, 255, 200])]
        whiteRange = [np.array([0, 0, 230]), np.array([180, 25, 255])]
        blueRange = [np.array([100,50,50]), np.array([130,255,255])]
        return [blackRange, silverRange, redRange, whiteRange, blueRange]
        
    def run(self):
        event_extraction_times = []
        while True:
            if not colourQueue.empty():

                frame, cars, predictions, frameNo = colourQueue.get()
                if frameNo == 0:
                    # No more frames
                    # Tell output no more frames
                    outputQueue.put((frame, cars, predictions, [], frameNo))
                    break

                if not cars:
                    # No cars detected
                    # Set extraction time to 0
                    event_extraction_times.append(0.0)
                    continue

                start_time_milli = int(time.time() * 1000)
                colours = self.classify_colour(cars)
                extraction_time_milli = int(time.time() * 1000) - start_time_milli
                event_extraction_times.append(extraction_time_milli)
                outputQueue.put((frame, cars, predictions, colours, frameNo))

        with open("Q3_extraction_times.csv", "a") as f:
            w = csv.writer(f)
            w.writerow(event_extraction_times)
                
    
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
        
### Class to write the output to the csv and creating images with bounding boxes and labels ###
class Output(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(Output,self).__init__()
        self.target = target
        self.name = name

    def run(self):
        color = 1
        font = cv2.FONT_HERSHEY_PLAIN

        while True:
            if not outputQueue.empty():
                frame, cars, predictions, colours, frameNo = outputQueue.get()
                if frameNo == 0:
                    # No more frames
                    break

                for c, prediction, colour in zip(cars, predictions, colours):
                    car, box = c
                    x,y,w,h = box
                    if prediction:
                        label = f'{Colour(colour).name} Sedan'
                    else:
                        label = f'{Colour(colour).name} Hatchback'
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y + 85), font, color, 3)
                
                filename = f'results/frame{frameNo}.png'
                cv2.imwrite(filename, frame)
            
                h = HATCHBACK.copy()
                s = SEDAN.copy()

                # looping through colours and predictions to create structures for
                # hatchback and sedan colours for comparison with ground truth
                for colour, prediction in zip(colours, predictions):
                    if prediction:
                        s[colour - 1] += 1
                    else:
                        h[colour - 1] += 1

                # frame no, sedan colours (5 cols), hatchback colours (5 colours), num. cars in frame
                data = [frameNo] + s + h + [len(predictions)]
                with open("predictions.csv", "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(data)

### Main stub to call all the classes ###
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