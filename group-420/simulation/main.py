import tensorflow as tf
from utils.config import config
import utils.dataloader as dataloader
import utils.model as models
import datetime

dataset = tf.data.Dataset.list_files(config["path"], shuffle=True, seed=420)\
    .map(dataloader.process_path, num_parallel_calls=tf.data.AUTOTUNE)\
    .filter(lambda _, __, speed: speed > 15)\
    .map(lambda i, a, s: (i, tf.clip_by_value(a, -config["angle"], config["angle"]), s))\
    .map(lambda i, a, _: (i, a), num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(config["batch_size"])\
    .prefetch(2)\
    .shuffle(config["batch_size"], reshuffle_each_iteration=True)

val_dataset = tf.data.Dataset.list_files(config["val_path"], shuffle=True, seed=420)\
    .map(dataloader.process_path, num_parallel_calls=tf.data.AUTOTUNE)\
    .filter(lambda _, __, speed: speed > 15)\
    .map(lambda i, a, s: (i, tf.clip_by_value(a, -config["angle"], config["angle"]), s))\
    .map(lambda i, a, _: (i, a), num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(config["batch_size"])\
    .prefetch(2)\
    .shuffle(config["batch_size"], reshuffle_each_iteration=True)

if config["saved_model_path"]:
    model = tf.keras.models.load_model(config["saved_model_path"])
    print("Retraining Saved Model")
else:
    model = models.Model420(config["crop"], config["angle"])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.build((config["batch_size"], 160, 320, 3))

# summary and logging
print(model.summary())
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(dataset, validation_data=val_dataset, epochs=config["epochs"], callbacks=[tensorboard_callback])
model.save("model", save_format="tf")
