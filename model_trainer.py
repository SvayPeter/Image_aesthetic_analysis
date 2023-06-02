import numpy as np
import argparse
from path import Path
import os
import cv2

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input


import keras.utils as image
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,ConfusionMatrixDisplay
import datetime


#load and preprocessing training data, return train and test data
def data_loader(IMAGE_SHAPE = (224, 224), BATCH_SIZE = 32):
    train_dir = "C:/Users/peter/Desktop/Code/Projects/Image_Aesthetic_Analysis_Project/datasets/car_dataset/Train"
    test_dir = "C:/Users/peter/Desktop/Code/Projects/Image_Aesthetic_Analysis_Project/datasets/car_dataset/Test/"

    train_datagen = ImageDataGenerator(rescale=1/255.,
                                    horizontal_flip=True, 
                                    vertical_flip=True,
                                    rotation_range=10,
                                    width_shift_range=0.2, 
                                    height_shift_range=0.2,
                                    zoom_range=0.2,)
    test_datagen = ImageDataGenerator(rescale=1/255.,
                                    horizontal_flip=True, 
                                    vertical_flip=True,
                                    rotation_range=10,
                                    width_shift_range=0.2, 
                                    height_shift_range=0.2,
                                    zoom_range=0.2,)

    print("Training images:")
    train_data = train_datagen.flow_from_directory(train_dir,
                                                target_size=IMAGE_SHAPE,
                                                batch_size=BATCH_SIZE,
                                                class_mode="categorical")

    print("Testing images:")
    test_data = test_datagen.flow_from_directory(test_dir,
                                                target_size=IMAGE_SHAPE,
                                                batch_size=BATCH_SIZE,
                                                class_mode="categorical")



    #y_test=np.concatenate([test_data .next()[1] for i in range(test_data .__len__())])
    #y_test=np.argmax(y_test, axis=1)
    return train_data, test_data

#callback function for writing logs
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

#Build model and return new model
def model_builder():
    from hyperopt import hp
    image_size = 224
    base_model = InceptionResNetV2(
        input_shape=(image_size, image_size, 3), 
        include_top=False, 
        pooling='avg', 
        weights=None)
    
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('weights/inception_resnet_weights.h5')

    base_inputs = model.layers[0].input
    base_outputs=  model.layers[-2].output

    #base_outputs = Dense(4, activation=tf.nn.relu, input_shape=(4,), kernel_regularizer=regularizers.l2(l=0.1))(base_outputs)
    #base_outputs = Dense(4, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(l=0.1))(base_outputs)
    output = Dense(2, activation='sigmoid')(base_outputs)

    new_model = Model(inputs = base_inputs, outputs = output)

    for layer in model.layers[:-40]:
        layer.trainable = False

   # hp_learning_rate = hp.choice('learning_rate', options=[1e-2, 1e-3, 1e-4])

    optimizer = Adam(learning_rate=1e-3)
    new_model.compile(optimizer, loss="binary_crossentropy", metrics=['accuracy'])
    new_model.summary()
    
    return new_model

#train the model and return history
def train_model(new_model, train_data, test_data, epochs):
    history = new_model.fit(train_data,
              epochs=epochs,
              steps_per_epoch=len(train_data),
              validation_data=test_data,
              validation_steps=len(test_data),
              shuffle=True,
              callbacks=[create_tensorboard_callback(dir_name="logs", experiment_name="test")])
    return history

#print accuracy and validation results
def show_accuracy_vs_val(history, epochs):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range=range(epochs)

    plt.figure(figsize=(15, 15))
    ax = plt.subplot(2, 2, 1)
    ax.set_ylim([0,1])
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    ax2 = plt.subplot(2, 2, 2)
    ax2.set_ylim([0,3])
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    return

#apply model to predict data
def predict(new_model, test_data):
    y_pred = new_model.predict(test_data)
    return y_pred,

#print confusion matrix and claassification report
def show_details(y_test,y_pred,target_names = ['Bad', 'Good']):
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test,y_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test,y_pred))
    disp.plot()
    plt.show()
    print(classification_report(y_test,y_pred, target_names=target_names))
