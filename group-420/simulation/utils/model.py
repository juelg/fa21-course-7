    
   
import tensorflow as tf
import tensorflow.keras as k


class Model420(k.Model):
    def __init__(self):
        super(Model420, self).__init__()
        self.conv0 = k.layers.Conv2D(filters=8, kernel_size=(5, 5), strides=4, activation="relu")
        self.conv1 = k.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=4, activation="relu")
        self.flatten = k.layers.Flatten()
        self.dropout = k.layers.Dropout(0.2)
        self.dense0 = k.layers.Dense(128, activation="relu")
        self.dense1 = k.layers.Dense(64, activation="relu")
        self.dense2 = k.layers.Dense(1, activation="relu")

    def call(self, inputs, training=False):
        output = self.conv0(inputs)
        output = self.conv1(output)
        output = self.flatten(output)
        output = self.dropout(output, training)
        output = self.dense0(output)
        output = self.dense1(output)
        output = self.dense2(output)
        return output
