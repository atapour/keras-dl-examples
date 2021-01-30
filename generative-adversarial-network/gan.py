# -*- coding: utf-8 -*-
"""
# Keras code for a Generative Adversarial Network using CIFAR-10
---

## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@newcastle.ac.uk

This notebook will provide an example for a Generative Adversarial Network (GAN), generating images from the CIFAR-10 dataset.

This is a code demonstration for CSC8637: Deep Learning module, Lecture 11: Generative Models. 

Copyright (c) 2021 School of Computing, Newcastle University, UK.

License : LGPL - http://www.gnu.org/licenses/lgpl.html

The CIFAR-10 dataset contains 50,000 training images and 10,000 test images. In the following snippet, we first load and explore the dataset by visualising the first few images within the dataset:
"""

# Required imports:
from matplotlib import pyplot
from tensorflow.keras.datasets import cifar10

# The dataset should contain 50,000 training images and 10,000 test images.
# Loading the dataset:
print('CIFAR-10 Dataset!')
(train_X, train_Y), (test_X, test_Y) = cifar10.load_data()

# Displaying the first thirty-six images within the dataset:
for i in range(36):
  # subplot:
  pyplot.subplot(6, 6, i+1)
  # plot image without any axes:
  pyplot.axis('off')
  pyplot.imshow(train_X[i])
# show the first 36 images:
pyplot.show()

# Dispalying information about the loaded dataset:
print(f'There are {train_X.shape[0]} images of size {train_X.shape[1:]} in the Training set.')
print(f'There are {test_X.shape[0]} images of size {test_X.shape[1:]} in the Test set.')

"""We will use the images in the training set to train our GAN model. The goal is for the model to capture the underlying distribution of the data and generate new samples from the distribution. Now we can begin preparing for the training process. First, let's import all that we need:"""

import sys
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.datasets.cifar10 import load_data
from matplotlib import pyplot
from tensorflow import keras
import tensorflow as tf

print('Keras version:', keras.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

"""Let's start by defining the architectures of our networks. Our model is *loosely* based on [DCGAN](https://arxiv.org/abs/1511.06434).

Our discriminator will receive an image of size 32x32x3 (size of CIFAR-10 data) and perform binary classification. All downsampling within the network will be done using strided convolutions:
"""

# Function used to create our discriminator network:
def create_discriminator(input_shape=(32,32,3)):

  model = Sequential()

  # At first, the discriminator will recieve the input image:
  model.add(Conv2D(64, (3,3), padding='same', input_shape=input_shape))
  model.add(LeakyReLU(alpha=0.2))

  # downsample the features by half:
  model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))

  # downsample the features by half:
  model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))

  # downsample the features by half:
  model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))

  # the feature map is now flattened and classified (binary - sigmoind, BCE):
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))

  # adding the optimiser and the loss function:
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

  # print model summary:
  print(model.summary())

  return model

print('Done!')

"""Our generator will receive a 100d latent vector as its input and will generate images of size 32x32x3. We will use transpose convolutions here to upsample the feature maps. The 100d noise vector will be mapped into a 4x4x256 feature vector and then upsampled from there.

The output layer of the generator has the activation function `tanh`. This means the output images will have pixel values in the range \[-1,1\]. So make sure your real samples (input images from the training dataset) are also in the same range. You could also use a `sigmoid` function, in which case the outputs are and the inputs need to be in the range \[0,1\].
"""

# Function used to create our generator network:
def create_generator(z_dim):

  model = Sequential()

  # map the z vector to a 4x4 image with 256 feature channels:
  # first we create a 4096d vector using a fully connected layer:
  starting_feature_map_size = 256 * 4 * 4
  model.add(Dense(starting_feature_map_size, input_dim=z_dim))
  model.add(LeakyReLU(alpha=0.2))

  # and then reshape the 4096d vector to a 4x4x256 feature map:
  model.add(Reshape((4, 4, 256)))

  # upsample the features to 8x8
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))

  # upsample the features to 16x16
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))

  # upsample the features to 32x32
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))

  # the output conv layer will create a 32x32x3 image:
  # Note that the output layer of the generator has an activation function 'tanh'. This means the output images will have pixel
  # values in the range [-1,1]. So make sure your real samples (input images from the training dataset) are also
  # in the same range. You could also use a 'sigmoid' function, in which case the input and the output need to be
  # in the range [0,1].
  model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))

  # Note that since the generator is trained using gradients from the discriminator, it is never trained alone.
  # So we don't need an optimiser in this function.

  # print model summary:
  print(model.summary())

  return model

print('Done!')

"""Now, every time we want to train the generator, it needs to be trained end to end with the discriminator so we need a combined generator-discriminator model:"""

# Function used to create our combined generator and discriminator model, for training the generator:
def create_gan(generator, discriminator):

  # since this model is only used to train the generator, we make our discriminator not trainable:
  discriminator.trainable = False

  # we then create an end-to-end model connecting the two networks:
  model = Sequential()
  # add the generator network
  model.add(generator)
  # add the discriminator network
  model.add(discriminator)

  # adding the optimiser and the loss function (BCE):
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)

  return model

print('Done!')

"""We will now define a series of functions that will deal with our data in various ways. First, let's define two functions. The first one defined below will load and prepare our training data from CIFAR-10 and the second will select "real samples" from the dataset to train the discriminator with:"""

# This function loads and processes CIFAR-10 images:
def load_prepare_dataset():

	# load the training dataset. We discard the test set and all the labels.
  # Generative models are unsupervised, if you remember from the lectures, and do not need labels:
  (train_x, _), (_, _) = load_data()

  # convert image pixel values to floating point values:
  x = train_x.astype('float32')

  # scale image pixel values from [0,255] to [-1,1] (remember the tanh used in the last layer of our generator)
  x = (x - 127.5) / 127.5

  return x

# ---------------------------------------------------
# ---------------------------------------------------

# This function selects and returns real samples from the already loaded and processed training set:
def get_real_samples(data, num_samples):

  # choose random instances from our dataset. We use numpy to get the random indexes:
  i = randint(0, data.shape[0], num_samples)

  # get the selected images from the dataset using the random indices:
  x = data[i]

  # since this function returns real samples, we need 'real' class labels for the discriminators.
  # This means these images should have the label "1":
  y = ones((num_samples, 1))

  return x, y

print('Done!')

"""Our generator will receive a 100d noise vector sampled from a "Normal Gaussian". We need a function to sample and return such a vector:"""

# Function to sample a noise vector from a Normal Gaussian for the generator input:
def sample_z(z_dim, num_samples):

  # sample a noise vector from a standard Gaussian
  z = randn(z_dim * num_samples)

  # reshape the noise vector into a batch of inputs for the generator:
  z = z.reshape(num_samples, z_dim)

  return z

print('Done!')

"""To train the discriminator, we also need fake samples that are generated by our generator network. So we need a function to generate fake samples using the generator. As these samples are fake, they should have the label '0' to properly train the discriminator:"""

# Function used to generate fake samples to train the discriminator:
def generate_fake_samples(generator, z_dim, num_samples):

  # to use the generator, we need to sample the noise vector z first:
  z = sample_z(z_dim, num_samples)

  # generate fake samples using the generator:
  x = generator.predict(z)

  # the samples are fake so they should have the class labels '0':
  y = zeros((num_samples, 1))

  return x, y

print('Done!')

"""Since we want to actually see (visually examine) the fruits of our labour (the generated images), the functions in the following snippets will enable us to use what we have trained to generate and plot some images:

"""

# Function used to plot the generated images:
def plot_image(images, epoch, n=3):

  # scale the images from [-1,1] to [0,1]
  images = (images + 1) / 2.0

  # loop to go through images and plot them
  for i in range(n * n):
    # subplot
    pyplot.subplot(n, n, 1 + i)
    # plot image without any axes:
    pyplot.axis('off')
    # plot the image pixels:
    pyplot.imshow(images[i])

  pyplot.show()

# ---------------------------------------------------
# ---------------------------------------------------

# Function to evaluate the model and plot results images:
def eval_model(epoch, generator, discriminator, data, z_dim, num_samples=150):

  # get real samples:
  x_real, y_real = get_real_samples(data, num_samples)

  # evaluate the discriminator network on real samples:
  _, real_acc = discriminator.evaluate(x_real, y_real, verbose=0)

  # get fake samples:
  x_fake, y_fake = generate_fake_samples(generator, z_dim, num_samples)

  # evaluate the discriminator network on fake examples:
  _, fake_acc = discriminator.evaluate(x_fake, y_fake, verbose=0)

  # summarise discriminator performance:
  print('\n---------------------------')
  if epoch == 0:
    print('This is what the generator images look like, before we actually train the GAN:')
  else:
    print(f'** End of Epoch {epoch+1}: Real Accuracy: {real_acc*100} - Fake Accuracy: {fake_acc*100}')

  # plot the generated images:
  plot_image(x_fake, epoch)
  print('---------------------------\n')

print('Done!')

"""Now that all preparations are made and the network architecture has been designed, it is time to start training the model.

First, let's decide the dimensions of the latent vector *z*:
"""

z_dim = 100

print('Done!')

"""Let's create the networks and set our batch size and number of epochs:"""

# discriminator:
discriminator = create_discriminator()

# generator:
generator = create_generator(z_dim)

# combined gan:
gan = create_gan(generator, discriminator)

# number of epochs and batch size:
num_epochs = 50
batch_size = 32

"""We are now ready to load the data:"""

# load image data
dataset = load_prepare_dataset()

"""Now for the main event, the **main training loop**.

We are going to iterate over the number of epochs and the batches in every epoch in a nested loop.

The losses and a number of sample images will be displayed after every 3 epochs of training so we can inspect the progress of our model:
"""

# loop over the number of epochs
for i in range(num_epochs):

  # every 5 epochs, print the losses and display images:
  if i % 3 == 0:
    eval_model(i, generator, discriminator, dataset, z_dim)

  # loop over the number of batches in every epoch
  for j in range(dataset.shape[0] // batch_size):

    # get randomly selected 'real' samples
    x_real, y_real = get_real_samples(dataset, batch_size)

    # train the discriminator using real samples:
    discriminator_loss_real, _ = discriminator.train_on_batch(x_real, y_real)

    # generate fake examples for training the discriminator:
    x_fake, y_fake = generate_fake_samples(generator, z_dim, batch_size)

    # train the discriminator using fake samples:
    discriminator_loss_fake, _ = discriminator.train_on_batch(x_fake, y_fake)

    # get the noise vector z for training the generator:
    z = sample_z(z_dim, batch_size)
    # now we need labels for training the generator. To train the generator, generated fake images are passed into the discriminator.
    # but the network will be given the 'real' label '1':
    y_generator = ones((batch_size, 1))

    # train the generator:
    generator_loss = gan.train_on_batch(z, y_generator)

    # print the loss every 100 steps:
    if (j) % 100 == 0:
      print(f'Epoch: {i+1:03} - Batch: {j:04}/{(dataset.shape[0]//batch_size)} - discriminator_loss_real={discriminator_loss_real:.2f} - discriminator_loss_fake={discriminator_loss_fake:.2f} - generator_loss={generator_loss:.2f}')

print('Done!')