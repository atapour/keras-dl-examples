# -*- coding: utf-8 -*-
"""
# Keras Code for a CNN-based Classification using CIFAR-10
---

## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@newcastle.ac.uk

This notebook will provide an example for a number of well-known convolutional neural network classifiers operating on CIFAR-10 data.

This is a code demonstration for CSC8637: Deep Learning module, Lecture 6: Convolutional Networks: Classification Architectures.

Copyright (c) 2021 School of Computing, Newcastle University, UK.

License : LGPL - http://www.gnu.org/licenses/lgpl.html

The CIFAR-10 dataset contains 50,000 training images and 10,000 test images. In the following snippet, we first load and explore the dataset by visualising the first few images within the dataset:
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
from tensorflow.keras.layers import GlobalAveragePooling2D
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

"""Now, we are ready to define our classification model. To enable experimenting on several networks, in the following, we add support for selecting what architecture we are going to use. Note that the choice of network is hard-coded here, but of course it would be better if it passed in as an argument (do as I say, not as I do):

"""

# These networks are supported:
network_names = [ 'mobilenet', 'resnet50', 'vgg16', 'vgg19', 'densenet121' ]

print("The following network architectures are available: ", network_names)

####################################
# The network id is selected here: #
network_id = 2
####################################

selected_network = network_names[network_id]
print(f"Selected network: {selected_network}")

"""Now, let's import the necessary architectures from Keras and set up functions that will prepare them for use. If the *weights* argument is set to None, the network will be trained from scratch. Here, it is set to *imagenet*, which means the models are all pre-trained on the imagenet dataset.

*We will talk about and experiment with this in a future lecture and code demonstration.*
"""

# Importing the models:
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.densenet import DenseNet121

# This function defines our classification network:
def create_model(selected_network):

  # The model is selected amongst the supported architectures:
  model_name = {
    'mobilenet'   : MobileNet,
    'resnet50'    : ResNet50,
    'vgg16'       : VGG16,
    'vgg19'       : VGG19,
    'densenet121' : DenseNet121
  }[selected_network]

  base_model = model_name(input_shape=(32,32,3), weights='imagenet', include_top=False)
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

"""Now that all preparations are made and the model has been designed, it is time to start training the model.

First, let's load the dataset:
"""

trainX, trainY, testX, testY = load_data()

"""Now, we pre-process the images:"""

trainX, testX = pre_process_data(trainX, testX)

"""Let's create the model:"""

model = create_model(selected_network)

"""The model can now be trained for 10 epochs with a batch size of 64:"""

history = model.fit(trainX, trainY, epochs=10, batch_size=64, validation_data=(testX, testY))
print('Done!')

"""After the training is complete, we can evaluate the model on the test set and obtain the final accuracy level:"""

_, acc = model.evaluate(testX, testY, verbose=1)
print('Accuracy: %.3f' % (acc * 100.0))

"""We can plot the loss and accuracy curves to better analyse the training process.

The **blue** curves indicate performance over the **training data** and the *red* curves represent model performance over the *test data*:
"""

plot_curves(history)