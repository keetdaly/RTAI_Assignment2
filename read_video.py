from keras.preprocessing import image
from keras.models import model_from_json
import cv2
import numpy as np

# Process image for MobileNet
def process_image(img):
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor

# Set up Yolo
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
with open('coco.names','r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Set up MobileNet model
json_file = open('model.json','r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)
model.load_weights('model.h5')

# Start video
video = 'video.mp4'
cap = cv2.VideoCapture(video)
frame_predictions = []

while(cap.isOpened()):
    print("Reading frame")
    # Read frame
    ret, frame = cap.read()
    
    
    if ret:
        height, width, channels = frame.shape
    
        # Resize image, process for input to Yolo
        f = cv2.resize(frame, None, fx=0.4, fy=0.4)
        blob = cv2.dnn.blobFromImage(f, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
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
                    car = frame[y:y+h, x:x+w]
                    car_tensor = process_image(car)
                    prediction = model.predict(car_tensor)
                    frame_predictions.append(prediction[0].tolist())
                else:
                    frame_predictions.append([0,0])
    else:
        break
cap.release()
cv2.destroyAllWindows()
with open("frame_predictions.txt","w") as f:
    for l in frame_predictions:
        f.write(str(l))
        f.write('\n')
