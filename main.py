import numpy as np
import pandas as pd 
import sklearn as sk
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import datasets
from tensorflow.keras import utils
from sklearn.datasets import load_files

# system data
train_dir = './images/train'
test_dir = './images/test'
valid_dir = './images/valid'

# images will be rescaled by 1./255
train_data_generation = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, 
                                                                        shear_range=0.2, 
                                                                        zoom_range=0.2, 
                                                                        horizontal_flip=True)

test_data_generation = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# images in batches of 10
# number of samples processed before the model is updated
train_generator = train_data_generation.flow_from_directory(train_dir, target_size=(200, 200), 
                                                            batch_size=10, class_mode='categorical')

validation_generator = test_data_generation.flow_from_directory(valid_dir, target_size=(200, 200), 
                                                                batch_size=10, class_mode='categorical')

# training a model with 98% accuracy
class get_Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.98 ):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True 

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (2,2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) 
print('Compiled!')

callbacks = get_Callback()
history = model.fit(train_generator, batch_size = 32, epochs=20, validation_data=validation_generator, callbacks=[callbacks], verbose=1, shuffle=True)

# plotting graphs
plt.figure(1)  
# history for accuracy  
plt.subplot(211)  
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])  
plt.title('models')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='lower right')

# history for loss    
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper right')  
plt.show()

def get_load_dataset(path):
    data = sk.datasets.load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    return files,targets,target_labels

x_test, y_test, target_labels = get_load_dataset(test_dir)

number_of_classes = len(np.unique(y_test))
number_of_classes

y_test = tf.keras.utils.to_categorical(y_test,number_of_classes)

def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(file)))
    return images_as_array


x_test = np.array(convert_image_to_array(x_test))
print('Test set shape : ',x_test.shape)

# visualize test prediction.
y_pred = model.predict(x_test)

# plot a raandom sample of test images, their predicted labels, and ground truth
figure = plt.figure(figsize=(16, 9))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    axis = figure.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    axis.imshow(np.squeeze((x_test[idx] * 255).astype(np.uint8)))
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_test[idx])
    axis.set_title("{} ({})".format(target_labels[pred_idx], target_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))