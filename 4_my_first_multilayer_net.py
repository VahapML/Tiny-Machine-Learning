import tensorflow as tf
import numpy as np
from tensorflow import keras


first_layer = keras.layers.Dense(units=4, input_shape=[1])
second_layer = keras.layers.Dense(units=2)
third_layer = keras.layers.Dense(units=1)

model = tf.keras.Sequential([first_layer, second_layer, third_layer])
model.compile(optimizer="sgd", loss="mean_squared_error")

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=200)

print(model.predict([10.0]))