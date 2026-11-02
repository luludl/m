import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

train = r"Downloads/Pneumonia X Ray-20221031T100017Z-001 (1)/Pneumonia X Ray/train"
test = r"Downloads/Pneumonia X Ray-20221031T100017Z-001 (1)/Pneumonia X Ray/test"

train_gen = ImageDataGenerator(rescale = 1./255).flow_from_directory(train,target_size = (224,224),batch_size = 32,class_mode = 'categorical')
test_gen = ImageDataGenerator(rescale = 1./255).flow_from_directory(test,target_size = (224,224),batch_size = 32,class_mode = 'categorical')


class_names = list(train_gen.class_indices.keys())

print(class_names)


model = Sequential([
    Conv2D(64,(3,3),activation = 'relu',input_shape = (224,224,3)),
    MaxPooling2D((2,2)),
    Conv2D(32,(3,3),activation = 'relu'),
    MaxPooling2D((2,2)),
    Conv2D(16,(3,3),activation = 'relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation = 'relu'),
    Dense(4,activation = 'softmax')

])

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

history = model.fit(train_gen,validation_data = test_gen,epochs = 5,verbose = 1)


x_test,y_test = next(test_gen)
preds = np.argmax(model.predict(x_test[:5]),axis = 1)
actual = np.argmax(y_test[:5],axis = 1)

plt.figure(figsize = (12,4))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_test[i])
    plt.title(f"p:{class_names[preds[i]]}\nA : {class_names[actual[i]]}")
    plt.axis('off')
plt.show()

print(history.history['accuracy'])
print(history.history['val_accuracy'])
