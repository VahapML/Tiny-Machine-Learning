import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

image = misc.ascent()

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(image)
plt.show()


feature_map = np.copy(image)
size_x = feature_map.shape[0]
size_y = feature_map.shape[1]

print("Image dimesions:")
print(size_x, size_y)

# This filter detects edges nicely
# It creates a convolution that only passes through sharp edges and straight
# lines.

# Experiment with different values for fun effects.
#filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]

# A couple more filters to try for fun!
# filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# If all the digits in the filter don't add up to 0 or 1, you 
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight  = 1

for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      convolution = 0.0
      convolution = convolution + (image[x - 1, y-1] * filter[0][0])
      convolution = convolution + (image[x, y-1] * filter[1][0])
      convolution = convolution + (image[x + 1, y-1] * filter[2][0])
      convolution = convolution + (image[x-1, y] * filter[0][1])
      convolution = convolution + (image[x, y] * filter[1][1])
      convolution = convolution + (image[x+1, y] * filter[2][1])
      convolution = convolution + (image[x-1, y+1] * filter[0][2])
      convolution = convolution + (image[x, y+1] * filter[1][2])
      convolution = convolution + (image[x+1, y+1] * filter[2][2])
      convolution = convolution * weight
      if(convolution<0):
        convolution=0
      if(convolution>255):
        convolution=255
      feature_map[x, y] = convolution


# Plot the image. Note the size of the axes -- they are 512 by 512
plt.gray()
plt.grid(False)
plt.imshow(feature_map)
#plt.axis('off')
plt.show()   


# Fashion MNIST Convolutional Neural Network
import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (val_images, val_labels) = mnist.load_data()


training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0

val_images = val_images.reshape(10000, 28, 28, 1)
val_images = val_images / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=tf.nn.relu, input_shape=(28,28,1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, validation_data=(val_images, val_labels), epochs=20)