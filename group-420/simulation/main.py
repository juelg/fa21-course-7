import tensorflow as tf
import utils.config as configurations
import utils.dataloader
import utils.dataloader as dataloader

config = configurations.local_config


def main():
    dataset = tf.data.Dataset.list_files(config["path"])\
        .map(dataloader.process_path)\
        .batch(config["batch_size"])\
        .prefetch(2)

    #list(dataset.as_numpy_iterator())
    for img, label in dataset:
        print(label.numpy())
        break

if __name__ == "__main__":
    main()
