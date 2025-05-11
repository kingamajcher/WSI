import tensorflow as tf
from tensorflow import keras

# loading data
(data_train, labels_train), (data_test, labels_test) = keras.datasets.mnist.load_data()

# normalising data from (0, 255) to (0.0, 1.0)
data_train = data_train / 255.0
data_test = data_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(data_train, labels_train, epochs=9)

model.summary()

model.evaluate(data_test, labels_test, verbose=2)

model.save("model.h5") 