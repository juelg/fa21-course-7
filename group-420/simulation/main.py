import tensorflow as tf
from utils.config import config
import utils.dataloader as dataloader
import utils.model as models

dataset = tf.data.Dataset.list_files(config["path"], shuffle=True, seed=420)\
    .map(dataloader.process_path, num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(config["batch_size"])\
    .prefetch(2)\
    .shuffle(config["batch_size"], reshuffle_each_iteration=True)

val_dataset = tf.data.Dataset.list_files(config["val_path"], shuffle=True, seed=420)\
    .map(dataloader.process_path, num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(config["batch_size"])\
    .prefetch(2)\
    .shuffle(config["batch_size"], reshuffle_each_iteration=True)

model = models.Model420()
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.build((config["batch_size"], 160, 320, 3))

print(model.summary())

model.fit(dataset, validation_data=val_dataset, epochs=config["epochs"])
model.save("model", save_format="tf")
