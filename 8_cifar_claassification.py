import tensorflow as tf
import matplotlib.pyplot as plt

from keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

FIRST_LAYER = layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", input_shape=(32,32,3))
HIDDEN_LAYER_1 = layers.MaxPooling2D(pool_size=(2,2))
HIDDEN_LAYER_2 = layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")
HIDDEN_LAYER_3 = layers.Conv2D(filters=32, kernel_size=(5,5), activation="relu")
HIDDEN_LAYER_4 = layers.MaxPooling2D(pool_size=(2,2))
HIDDEN_LAYER_5 = layers.Dense(units=16, activation="relu")
LAST_LAYER = layers.Dense(units=10, activation="softmax")

model = models.Sequential([ FIRST_LAYER,
                            HIDDEN_LAYER_1,
                            HIDDEN_LAYER_2,
                            HIDDEN_LAYER_3,
                            HIDDEN_LAYER_4,
                            layers.Flatten(),
                            HIDDEN_LAYER_5,
                            LAST_LAYER])


LOSS = "sparse_categorical_crossentropy"
NUMBER_EPOCHS = 15

model.compile(optimizer="SGD", loss=LOSS, metrics=["accuracy"])
training_data = model.fit(train_images, train_labels, batch_size=100, epochs=NUMBER_EPOCHS, validation_data=(test_images, test_labels))

plt.plot(training_data.history["accuracy"])
plt.plot(training_data.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs -->")
plt.legend(['Train', 'Test'], loc='upper left')
plt.xlim([0, NUMBER_EPOCHS])
plt.ylim([0, 1.0])
plt.show()



model.compile(optimizer="Adam", loss=LOSS, metrics=["accuracy"])
training_data = model.fit(train_images, train_labels, batch_size=100, epochs=NUMBER_EPOCHS, validation_data=(test_images, test_labels))

plt.plot(training_data.history["accuracy"])
plt.plot(training_data.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs -->")
plt.legend(['Train', 'Test'], loc='upper left')
plt.xlim([0, NUMBER_EPOCHS])
plt.ylim([0, 1.0])
plt.show()