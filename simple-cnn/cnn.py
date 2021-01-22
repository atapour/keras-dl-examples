# -*- coding: utf-8 -*-
"""
# Keras code for a Convolutional Neural Network using CIFAR-10
---

## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@newcastle.ac.uk

This notebook will provide a simple example for a Convolutional Neural Network classifying CIFAR-10 data.

This is a code demonstration for CSC8637: Deep Learning module, Lecture 5: Convolutional Neural Networks.

Copyright (c) 2021 School of Computing, Newcastle University, UK.

License : LGPL - http://www.gnu.org/licenses/lgpl.html

The **CIFAR-10** dataset contains 50,000 training images and 10,000 test images. In the following snippet, we first load and explore the dataset by visualising the first few images within the dataset:
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

"""Now that the dataset has been verified, we can begin preparing for the training process. First, let's import all that we need:"""

from matplotlib import pyplot
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
import tensorflow as tf

print('Keras version:', keras.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

"""We now define two functions that will load and process the data in preparation for training:"""

# Function used to load the dataset:
def load_data():

  # Loading the built-in CIFAR-10 dataset from Keras:
  (train_X, train_Y), (test_X, test_Y) = cifar10.load_data()

  print('The dataset has been loaded:')
  print(f'Train: X={train_X.shape}, Y={train_Y.shape}')
  print(f'Test: X={test_X.shape}, Y={test_Y.shape}')

  # Converting the class labels to one hot vectors:
  train_Y = to_categorical(train_Y)
  test_Y = to_categorical(test_Y)

  return train_X, train_Y, test_X, test_Y
 
# -----------------------------

# Function used to prepare the data:
def pre_process_data(train_data, test_data):

  # Casting pixel values to floats:
  train_data = train_data.astype('float32')
  test_data = test_data.astype('float32')

  # normalising pixel values to range [0-1]
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

"""Now we are ready to design our model architecture.

The architecture comprises **three CONV layers** with **RELU activation functions**, each followed by **Max Pooling** layers. At the end, there is a **fully-connected** classifier that will classify the input into one of 10 outputs, using **cross entropy** as the loss function:
"""

# This function defines our neural network:
def create_model():

  model = Sequential()

  # The first conv layer with 32 kernels of 3*3 receiving an input of 32*32*3:
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))

  # Max pooling layer with a kernel of 2*2 and a stride of 2:
  model.add(MaxPooling2D((2, 2)))

  # Conv layer with 64 kernels of 3*3:
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

  # Max pooling layer with a kernel of 2*2 and a stride of 2:
  model.add(MaxPooling2D((2, 2)))

  # Conv layer with 128 kernels of 3*3:
  model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

  # Max pooling layer with a kernel of 2*2 and a stride of 2:
  model.add(MaxPooling2D((2, 2)))

  # The feature maps are flattened at this point to be passed into fully-connected layers:
  model.add(Flatten())

  # Fully-connected layers leading to 10 classes with a softmax activation function:
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(10, activation='softmax'))

  # The optimiser is stochastic gradient descent with a learning rate of f 0.001 and a momentum of 0.9:
  optim = SGD(lr=0.001, momentum=0.9)

  # The model optimises cross entropy as its loss function and will monitor classification accuracy:
  model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

  # Printing model summary:
  print(model.summary())

  return model

print('Done!')

"""Now that all preparations are made and the model has been designed, it is time to start training the model.

First, let's load the dataset:
"""

trainX, trainY, testX, testY = load_data()

"""Now, we pre-process the images using the function we defined earlier:"""

trainX, testX = pre_process_data(trainX, testX)

"""Let's create the model:"""

model = create_model()

"""The model can now be trained for 20 epochs with a batch size of 64:"""

history = model.fit(trainX, trainY, epochs=20, batch_size=64, validation_data=(testX, testY))
print('Done!')

"""After the training is complete, we can evaluate the model on the test set and obtain the final accuracy level:"""

_, acc = model.evaluate(testX, testY, verbose=1)
print('Accuracy: %.3f' % (acc * 100.0))

"""We can plot the loss and accuracy curves to better analyse the training process.

The **blue** curves indicate performance over the **training data** and the *red* curves represent model performance over the *test data*:
"""

plot_curves(history)