import tensorflow as tf
import tensorflow.keras as k

class Model420(k.Model):
    def __init__(self, crop, angle):
        super(Model420, self).__init__()

        self.angle = angle
        self.crop = k.layers.Cropping2D(cropping=((crop, 0), (0, 0)))
        
        self.conv0 = k.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=1, activation="relu")
        self.max0 = k.layers.MaxPool2D()
        self.bn0 = k.layers.BatchNormalization()

        self.conv1 = k.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation="relu")
        self.max1 = k.layers.MaxPool2D()
        self.bn1 = k.layers.BatchNormalization()

        self.conv2 = k.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu")
        self.max2 = k.layers.MaxPool2D()
        self.bn2 = k.layers.BatchNormalization()

        self.conv3 = k.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation="relu")
        self.max3 = k.layers.MaxPool2D()
        self.bn3 = k.layers.BatchNormalization()

        self.conv4 = k.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation="relu")
        self.max4 = k.layers.MaxPool2D()
        self.bn4 = k.layers.BatchNormalization()

        self.flatten = k.layers.Flatten()
        self.dropout = k.layers.Dropout(0.2)
        self.dense0 = k.layers.Dense(128, activation="relu")
        self.dense1 = k.layers.Dense(64, activation="relu")
        self.dense2 = k.layers.Dense(1, activation='tanh')

    def call(self, inputs, training=False):
        output = self.crop(inputs)
        
        output = self.conv0(output)
        output = self.max0(output)
        output = self.bn0(output, training=training)

        output = self.conv1(output)
        output = self.max1(output)
        output = self.bn1(output, training=training)

        output = self.conv2(output)
        output = self.max2(output)
        output = self.bn2(output, training=training)

        output = self.conv3(output)
        output = self.max3(output)
        output = self.bn3(output, training=training)

        output = self.conv4(output)
        output = self.max4(output)
        output = self.bn4(output, training=training)

        output = self.flatten(output)
        output = self.dropout(output, training)
        output = self.dense0(output)
        output = self.dense1(output)
        output = self.dense2(output)

        return output * self.angle
