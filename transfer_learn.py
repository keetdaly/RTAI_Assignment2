from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input

# Module for training the transfer learning model for car type classification
base_model = MobileNet(weights='imagenet', include_top=False)

# Defining layers for the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x) # adding dense layers so that the model can learn complex functions and for better classification results
x = Dense(1024,activation='relu')(x) #dense layer 2
x = Dense(512,activation='relu')(x) #dense layer 3
preds = Dense(2,activation='softmax')(x) #final layer with softmax activation

model = Model(inputs=base_model.input,outputs=preds)
for layer in model.layers[:28]:
    layer.trainable = False

# Reading the images for the type classification
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
train_generator = train_datagen.flow_from_directory('cars/images', target_size=(224,224), color_mode='rgb', batch_size=128,
                                                    class_mode='categorical', shuffle=True)

# Compiling the model with Adam optimizer, loss function as categorical cross entropy
# and evaluation metric as accuracy
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Training the model
step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=10)

# Saving the model as json object
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model")
