# -*- coding: utf-8 -*-
"""
# Keras Code for Visualising Activation Maps
---

## Author : Amir Atapour-Abarghouei, amir.atapour-abarghouei@newcastle.ac.uk

This notebook will provide an example for visualising activation maps from different blocks of a VGG-16.

This is a code demonstration for CSC8637: Deep Learning module, Lecture 12: Ethics and Challenges.

Copyright (c) 2021 School of Computing, Newcastle University, UK.

License : LGPL - http://www.gnu.org/licenses/lgpl.html

In this demo, we are going to visualise the activation maps from different layers of a VGG-16 network trained on ImageNet. The same process can of course be done for architectures other than VGG-16.

Let's start by importing what we need:
"""

# Required imports:
from PIL import Image, ImageEnhance
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
from skimage import io
from skimage.transform import resize
from tensorflow import keras
import tensorflow as tf

print('Keras version:', keras.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

"""First, we need an image to pass through our network. To make the process easier, we will get an image from the internet, but you can do this with any image your heart desires. """

# URL to download the image. Make sure this is a link to actual image (with the extension at the end):
url = 'https://images.freeimages.com/images/large-previews/006/young-dachshund-1362378.jpg'

# Read the image using skimage:
image = io.imread(url)

# Resize the image to 224x224, since this is what most CNNs expect. You might need to change the size depending on the network you use:
image = resize(image, (224, 224), anti_aliasing=True)

# Display the image:
pyplot.axis('off')
pyplot.imshow(image) 
print(f'Image with size {image.shape} has been loaded.')

"""We now need to process the image so it can accepted by our neural network. Here, we will use the function built into Keras:"""

# Convert the image to an array:
image = img_to_array(image)
# The model expects a 4D tensor so expand dimensions:
image = expand_dims(image, axis=0)
# Scale pixel values:
image = preprocess_input(image)

print(f'Done!')

"""We need to define a function that displays the activation maps we extract from the layer we want in a square grid:"""

# Function to display the activation maps:
def display(feature_maps, index, square):
  # Get the feature maps for the layer we are interested in:
  fmap = feature_maps[index]
  # Plot 2^square feature maps
  ix = 1

  for _ in range(square):
    for _ in range(square):
      # Create subplots:
      ax = pyplot.subplot(square, square, ix)
      # Turn off axis:
      ax.set_xticks([])
      ax.set_yticks([])
      # Extract and enhance filter channel using PIL
      img = fmap[0, :, :, ix-1]
      # Convert image to PIL:
      im = Image.fromarray(img)
      # Convert image to 3-channel for the enhancer:
      im = im.convert('RGB')
      # Create enhancer:
      enhancer = ImageEnhance.Contrast(im)
      # Increase the contrast by:
      factor = 10
      # Enhance (increase contrast):
      im_output = enhancer.enhance(factor)
      # Convert the image back to array:
      img = np.array(im_output)
      # Plot the images:
      pyplot.imshow(img[:,:,0], cmap='plasma')
      ix += 1
  # Show figure
  pyplot.show()

print(f'Done!')

"""Now, let's choose our model architecture. We are going to use VGG-16 here:"""

model = VGG16()
model.summary()

"""We are going to visulaise feature maps from each convolutional block within VGG16. We will do this by passing the image through the image once and outputting the feature maps from the layers we want.

The convolutional layers at the end of each of the 5 blocks are where we extract our feature maps. The indices of these layers are [2, 5, 9, 13, 17]. Note that this only applies to VGG16.

We will have to re-define the network to output the features we are looking for:

"""

# Re-define the network to output feature maps at the chosen layers:
layers = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in layers]
model = Model(inputs=model.inputs, outputs=outputs)
model.summary()

"""Now, let's pass the image through the network:"""

feature_maps = model.predict(image)
print('done!')

"""It is now time to display what we got. First, let's look at the output of the first 16 filters of the first block:"""

display(feature_maps=feature_maps, index=0, square=4)

"""Now, the second convolutional block:"""

display(feature_maps=feature_maps, index=1, square=4)

"""The third:"""

display(feature_maps=feature_maps, index=2, square=4)

"""Fourth:"""

display(feature_maps=feature_maps, index=3, square=4)

"""And now the last one:"""

display(feature_maps=feature_maps, index=4, square=4)

"""As you see, the deeper into the network we go, the harder it gets to actually see the maps."""