from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import os

# Module for testing the transfer learning model for car type classification
def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    return img_tensor

# Loading the trained model
json_file = open('model.json','r')
loaded_model = json_file.read()
json_file.close()
model = model_from_json(loaded_model)
model.load_weights('model.h5')

# Compiling the model
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
c1 = 0
c2 = 0

# Testing the model on the test set of each car type
for filename in os.listdir('cars/images/sedan'):
    img = load_image('cars/images/sedan/' + filename)
    pred = model.predict(img)
    if(pred[0][1] > 0.9):
        c1 += 1
    print("Sedan", pred)
    
print("Correct",c1)
for filename in os.listdir('cars/images/hatchback'):
    img = load_image('cars/images/hatchback/' + filename)
    pred = model.predict(img)
    if(pred[0][0] > 0.9):
        c2 += 1
    print("Hatchback", pred)
print("Correct", c2)
