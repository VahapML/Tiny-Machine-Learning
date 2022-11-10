import tensorflow as tf
import numpy as np

# load fashion mnist dataset from tensorflow dataset library
fashion_mnist = tf.keras.datasets.fashion_mnist
# split data into training and test sets
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

# build neural network model with 3 layers 
# and define input shape to be 28 x 28
# use softmax activation function to output class probabilities
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(units=512, activation = tf.nn.relu),
    tf.keras.layers.Dense(units=512, activation = tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

# normalize grayscale images
training_images = training_images / 255.0
test_images = test_images / 255.0

# choose optimizer, loss and metrics
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# train model with given training dataset and number of epochs
model.fit(training_images, training_labels, epochs=7)

print("Evaluation Results")
# evaluate model on test set
model.evaluate(test_images, test_labels)

# make predictions
classifications = model.predict(test_images)

# print and compare prediction results with actual labels
print(type(classifications))
print(classifications.shape)

print("Model Prediction Probablities for Image 1, 2, 3")
print(classifications[0])
print(classifications[1])
print(classifications[2])

print("Max Prediction Probablities for Image 1, 2, 3")

print(np.argmax(classifications[0]))
print(np.argmax(classifications[1]))
print(np.argmax(classifications[2]))

print("Expected Labels for Image 1, 2, 3")
print(test_labels[0])
print(test_labels[1])
print(test_labels[2])

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get("accuracy") > 0.88 ):
            self.model.stop_training = True



# build neural network model with 3 layers 
# and define input shape to be 28 x 28
# use softmax activation function to output class probabilities
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(units=512, activation = tf.nn.relu),
    tf.keras.layers.Dense(units=512, activation = tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

# choose optimizer, loss and metrics
model.compile(optimizer = tf.keras.optimizers.Adam(),
      loss = 'sparse_categorical_crossentropy',
      metrics=['accuracy'])

# create a callback instance from myCallback class
callback = myCallback()
model.fit(training_images, training_labels, epochs=7, callbacks=[callback])

print("Evaluation Results")
# evaluate model on test set
model.evaluate(test_images, test_labels)