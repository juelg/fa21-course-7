import tensorflow as tf
from utils.config import config
import utils.dataloader as dataloader
import utils.model as models
import datetime

dataset = tf.data.Dataset.list_files(config["path"], shuffle=True, seed=420)\
    .map(dataloader.process_path, num_parallel_calls=tf.data.AUTOTUNE)\
    .filter(lambda img, steering_angle, speed: speed > 15)\
    .map(lambda img, steering_angle, _: (img, steering_angle), num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(config["batch_size"])\
    .prefetch(2)\
    .shuffle(config["batch_size"], reshuffle_each_iteration=True)

val_dataset = tf.data.Dataset.list_files(config["val_path"], shuffle=True, seed=420)\
    .map(dataloader.process_path, num_parallel_calls=tf.data.AUTOTUNE)\
    .filter(lambda img, steering_angle, speed: speed > 15)\
    .map(lambda img, steering_angle, _: (img, steering_angle), num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(config["batch_size"])\
    .prefetch(2)\
    .shuffle(config["batch_size"], reshuffle_each_iteration=True)

model = models.Model420(config["crop"])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.build((config["batch_size"], 160, 320, 3))

# summary and logging
print(model.summary())
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(dataset, validation_data=val_dataset, epochs=config["epochs"],callbacks=[tensorboard_callback])
model.save("model", save_format="tf")
