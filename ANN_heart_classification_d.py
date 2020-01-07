import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers

tf.__version__

dataframe = pd.read_csv('heart.csv')

dataframe.head(10)

dataframe.shape

x = dataframe.drop(['target'], axis=1)
y = dataframe['target']

x.head()

y.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.4)

print(x_train.shape)
print(x_test.shape)

from tensorflow.keras import layers, model


def build_model():
    model = model.Sequesntial()
    model.add(layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model


model = build_model()

hist = model.fit(x_train, y_train, epochs=20, batch_size=10, validation_split=.1, verbose=1)