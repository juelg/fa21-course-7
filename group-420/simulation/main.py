import tensorflow as tf
import utils.config as configurations
import utils.dataloader as dataloader

config = configurations.local_config


def main():
    # TODO: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
    # buffersize
    dataset = tf.data.Dataset.list_files(config["path"], shuffle=True, seed=420)\
        .map(dataloader.process_path, num_parallel_calls=tf.data.AUTOTUNE)\
        .batch(config["batch_size"])\
        .prefetch(2)\
        .shuffle(3 * config["path"], reshuffle_each_iteration=True)


    print(list(dataset.as_numpy_iterator()))
    for img, label in dataset:
        print(label.numpy())
        break

if __name__ == "__main__":
    main()
