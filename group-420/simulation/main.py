import tensorflow as tf
import utils.config as configurations
import utils.dataloader
import utils.dataloader as dataloader

config = configurations.local_config


def main():
    print(dataloader.process_path(r"C:\Users\plain\Downloads\recording\out\center\22_4114_-200.jpg"))
    dataset = tf.data.Dataset.list_files(config["path"])\
        .map(dataloader.process_path)\
        .batch(config["batch_size"])\
        .prefetch(2)

    print(list(dataset.as_numpy_iterator()))


if __name__ == "__main__":
    main()
