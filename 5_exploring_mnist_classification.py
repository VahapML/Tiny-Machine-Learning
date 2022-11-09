import tensorflow as tf
import numpy as np

data = tf.keras.datasets.mnist

(training_images, training_labels), (val_images, val_labels) = data.load_data()

training_shape = training_images.shape
val_shape = val_images.shape

print("Training images shape: " + str(training_shape))
print("Validation images shape: " + str(val_shape))

training_images = training_images / 255.0
val_images = val_images / 255.0


model = tf.keras.models.Sequential(
    [tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(units=20, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)]
)

model.compile(optimizer="adam",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"])

model.fit(training_images, training_labels, epochs=10, validation_data=(val_images, val_labels))

model.evaluate(val_images, val_labels)

classifications = model.predict(val_images)

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
print(val_labels[0])
print(val_labels[1])
print(val_labels[2])