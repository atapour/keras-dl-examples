# -*- coding: utf-8 -*-
"""
# Keras code to demonstrate better training of neural networks using CIFAR-10
---

## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@newcastle.ac.uk

This notebook will provide an example to show how data augmentation and transfer learning can improve the performance of neural networks and prevent overfitting using the CIFAR-10 dataset.

This is a code demonstration for CSC8637: Deep Learning module, Lecture 10: Better Training of Neural Networks.

Copyright (c) 2021 School of Computing, Newcastle University, UK.

License : LGPL - http://www.gnu.org/licenses/lgpl.html

As with our other code demonstrations in our lecture series, we will be using Keras to train a model on the **CIFAR-10** dataset.

This dataset contains 50,000 training images and 10,000 test images. In the following snippet, we first load and explore the dataset by visualising the first few images within the dataset:
"""

# Required imports:
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# The dataset contains 50,000 training images and 10,000 test images.
# Loading the dataset:
print('CIFAR-10 Dataset!')
(train_X, train_Y), (test_X, test_Y) = cifar10.load_data()

# CIFAR-10 contains these classes:
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'lorry']

# -----------------------------
# This function will display the first 16 images of the dataset with their labels:
def visualize_data(train_X, train_Y, class_names):

  for i in range(16):
    # create subplot:
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    # plot image with the class name on the x-axis:
    plt.imshow(train_X[i])
    plt.xlabel(class_names[train_Y[i].item()])

  # adjust the subplots and show the first 16 images:
  plt.subplots_adjust(left=0.125,
                      bottom=0.1, 
                      right=0.9, 
                      top=0.9, 
                      wspace=0.2, 
                      hspace=0.35)
  plt.show()
# -----------------------------

# Displaying the first sixteen images within the dataset:
visualize_data(train_X, train_Y, class_names)

# Printing information about the loaded dataset:
print(f'There are {train_X.shape[0]} images of size {train_X.shape[1:]} in the Training set of the CIFAR-10 Dataset.')
print(f'There are {test_X.shape[0]} images of size {test_X.shape[1:]} in the Test set of the CIFAR-10 Dataset.')

"""In this demo, we want to demonstrate how we can prevent overfitting. As overfitting often happens when we are dealing with a small dataset, we are only going to use 5,000 of the training image in CIFAR-10:"""

train_X, train_Y = train_X[:5000], train_Y[:5000]

print(f'We have selected the first {train_X.shape[0]} images of size {train_X.shape[1:]} for our training dataset.')

"""Now that we have had a look at the dataset we will be working with, we can begin preparing for the training process. First, let's import all that we need:"""

from matplotlib import pyplot
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow import keras
import tensorflow as tf

print('Keras version:', keras.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

"""Since we are going to be running experiments here, we would like our results to be reproducible. So we need to make our setup as deterministic as possible. One thing that will help with this is setting seeds for any and all randomness in our code.

Note that especially since we are running our code on **GPU** cloud services, there are variables that we can't always control (easily) so reaching absolute determinism won't really be possible here. But setting seeds is always good practice.

Additionally, by setting seeds, we can control the stochasticity of our algorithm (for example, the effects of the random initialisation of our network weights). One can run a given experiment a number of times with different random seeds and then average the performance metrics across all the runs.

Note that different packages and libraries might have different random number generation processes. Here, we will pick a seed value of 0 for Numpy, TensorFlow, the Python Random module:
"""

import numpy as np
import tensorflow as tf
import random as rn

seed_value = 0

np.random.seed(seed_value)
rn.seed(seed_value)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.compat.v1.set_random_seed(seed_value)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

print('Done!')

"""Let's first choose the architecture we are going to use. We are going to use VGG16. In our first experiment, the model is **not pre-trained** and will use **no data augmentation**:"""

# This function defines our classification network:
def create_model():

  base_model = VGG16(input_shape=(32,32,3), weights=None, include_top=False)
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  output = keras.layers.Dense(10, activation='softmax')(x)
  model = keras.models.Model(inputs=[base_model.input], outputs=[output])

  # The optimiser is stochastic gradient descent with a learning rate of 0.01 and a momentum of 0.9:
  optim = SGD(lr=0.01, momentum=0.9)

  # The model optimises cross entropy as its loss function and will monitor classification accuracy:
  model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

  # Printing model summary:
  print(model.summary())

  return model

print('Done!')

"""We now define two functions that will load and process the data in preparation for training. Remember that we are only going to use 5,000 images for training:"""

# Function used to load the dataset:
def load_data():

  # Loading the built-in CIFAR-10 dataset from Keras:
  (train_X, train_Y), (test_X, test_Y) = cifar10.load_data()

  # We are going to use only 5,000 images for training:
  train_X, train_Y = train_X[:5000], train_Y[:5000]

  print('The dataset has been loaded:')
  print(f'Train: X={train_X.shape}, Y={train_Y.shape}')
  print(f'Test: X={test_X.shape}, Y={test_Y.shape}')

  # Converting class labels to one hot vectors:
  train_Y = to_categorical(train_Y)
  test_Y = to_categorical(test_Y)

  return train_X, train_Y, test_X, test_Y
 
#  ---------------------------------------

# Function used to pre-process the data:
def pre_process_data(train_data, test_data):

  # Casting pixel values to floats:
  train_data = train_data.astype('float32')
  test_data = test_data.astype('float32')

  # Normalising pixel values to range [0-1]
  train_data = train_data / 255.0
  test_data = test_data / 255.0

  print(f'Train data is in range {train_data.min()} to {train_data.max()}.')
  print(f'Test data is in range {test_data.min()} to {test_data.max()}.')

  return train_data, test_data

print('Done!')

"""We will also define a function responsible for plotting the curves:"""

# Function used to plot the curves for loss and accuracy:
def plot_curves(history):

  # Plotting the loss curve:
  plt.subplot(211)
  plt.title('Cross Entropy')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  # Plotting the training loss (blue):
  plt.plot(history.history['loss'], color='blue', label='train')
  # Plotting the test loss (red):
  plt.plot(history.history['val_loss'], color='red', label='test')
  # Legend for the plot:
  plt.legend(['train', 'test'], loc='upper left')

  # Plotting the accuracy curve:
  plt.subplot(212)
  plt.title('Classification Accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  # Plotting the training accuracy (blue):
  plt.plot(history.history['accuracy'], color='blue', label='train')
  # Plotting the test accuracy (red):
  plt.plot(history.history['val_accuracy'], color='red', label='test')
  # Legend for the plot:
  plt.legend(['train', 'test'], loc='upper left')

  plt.subplots_adjust(top=3)
  plt.show()

print('Done!')

"""Now that all preparations are made, it is time to start training the model.

First, let's load the dataset. Note that the size of our training data is very small (only 5,000 images and half the size of our test data), a situation ripe for overfitting:
"""

train_X, train_Y, test_X, test_Y = load_data()

"""Now, we pre-process the images:"""

train_X, test_X = pre_process_data(train_X, test_X)

"""Let's create the model:"""

model = create_model()

"""We are now going to train the model for 50 epochs (more than what we need with respect to the size of our training set) with a batch size of 64:"""

history = model.fit(train_X, train_Y, epochs=50, batch_size=64, validation_data=(test_X, test_Y))

"""Now that the training is done, we can evaluate the model on the test set and see what final accuracy levels we get:"""

_, acc = model.evaluate(test_X, test_Y, verbose=1)
print('Accuracy: %.3f' % (acc * 100.0))

"""Plotting the loss and accuracy curves will help us better analyse the training process.

The **blue** curves indicate performance over the **training data** and the *red* curves represent model performance over the *test data*:
"""

plot_curves(history)

"""As we can see, the performance is clearly very poor. Additionally, our Classification Accuracy curves are indicating that the model has overfit to the training data. We have thus established our **baseline model**.

Now, let's see how we can start improving the performance of the model.

Since the training dataset we are using here is very small (half of what we have in our test set), **Data Augmentation** can really help us.

All modern deep learning frameworks have common data augmentation techniques built in that can make our work easier. Keras is no exception. The [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) class will enable us to incorporate data augmentation into our training pipeline.

We are going to use the following image transformations:

*   Rotation
*   Horizontal Flip
*   Horizontal Translation
*   Vertical Translation

Note that the type of augmentations and amounts by which the images are transformed are *hyper-parameters* that may need to be tuned to the application and the type of data we are using by means of some search algorithm in order to achieve optimal performance:
"""

# Required import:
from keras.preprocessing.image import ImageDataGenerator

# Load and pre-process the data:
train_X, train_Y, test_X, test_Y = load_data()
train_X, test_X = pre_process_data(train_X, test_X)

# Create the Image Data Generator for our augmentation:
data_generator = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    )

# Create an iterator over the data:
data_iterator = data_generator.flow(train_X, train_Y, batch_size=64)

"""Let's have a look at our images post augmentation to see what they look like:"""

# -----------------------------
# Function that will display 16 augmented images of the dataset with their labels:
def view_augmented_data(train_X, train_Y, class_names):
  for i in range(16):
    # subplot:
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    # plot image:
    plt.imshow(train_X[i])
    class_index = train_Y[i].argmax()
    plt.xlabel(class_names[class_index])

  # show the images:
  plt.subplots_adjust(left=0.125,
                      bottom=0.1, 
                      right=0.9, 
                      top=0.9, 
                      wspace=0.2, 
                      hspace=0.35)
  plt.show()
# -----------------------------

batch_X, batch_Y = next(data_iterator)
view_augmented_data(batch_X, batch_Y, class_names)

"""We are using the [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) to perform our data augmentation.

We will not change the model since our last experiment that established our baseline and will only re-initialise the model just as we did in our last experiment.

This is because we want to see how data augmentation can help us improve the performance of our model and prevent overfitting.

So, let's train our model **with data augmentation**:
"""

# Create the model:
model = create_model()

# Train the model for 50 epochs with a batch size of 64:
history = model.fit(data_iterator, epochs=50, validation_data=(test_X, test_Y), verbose=1)

"""The training is complete. So we can now evaluate the model and take a look at our performance:"""

_, acc = model.evaluate(test_X, test_Y, verbose=1)
print('Accuracy: %.3f' % (acc * 100.0))

plot_curves(history)

"""We can see how much the model has improved. The classification accuracy curves for our training and test phases are much closer to each other and the final test accuracy level is significantly higher, all thanks to data augmentation.

But we can do even better!!!

At this point, we can go with **Transfer Learning** to see how it will affect things.

To keep things simple, we are going to load a VGG16 architecture with weights pre-trained on [ImageNet](http://www.image-net.org/) and then fine-tune the whole network (the final classification layer will be trained completely from scratch):
"""

# This function creates our classification network, this time pre-trained on ImageNet:
def create_model():

  # Keras can already provide us with a pre-trained network just by setting the weights argument to imagenet:
  base_model = VGG16(input_shape=(32,32,3), weights='imagenet', include_top=False)
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  output = keras.layers.Dense(10, activation='softmax')(x)
  model = keras.models.Model(inputs=[base_model.input], outputs=[output])

  # The optimiser is stochastic gradient descent with a learning rate of 0.01 and a momentum of 0.9:
  optim = SGD(lr=0.01, momentum=0.9)

  # The model optimises cross entropy as its loss function and will monitor classification accuracy:
  model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

  # Printing model summary:
  print(model.summary())

  return model

print('Done!')

"""Now, let's create and train the model. We will keep the same data augmentation from the last experiment in place, so the only change here is that the model has been **pre-trained on ImageNet**:"""

# Create the model:
model = create_model()

# Train the model for 50 epochs with a batch size of 64:
history = model.fit(data_iterator, epochs=50, validation_data=(test_X, test_Y), verbose=1)

"""After the training is complete, we can evaluate the model on the test set and plot the curves as we did before to see if we get any improvements:"""

_, acc = model.evaluate(test_X, test_Y, verbose=1)
print('Accuracy: %.3f' % (acc * 100.0))

plot_curves(history)

"""We can clearly see that the results have significantly improved. Transfer learning is an excellent idea and even in its simplest form is always recommended for most applications.

ImageNet is an extremely useful dataset, and most cutting-edge state-of-the-art CNNs for different applications and tasks are often pre-trained on this dataset.
"""