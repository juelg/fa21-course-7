import tensorflow as tf
import utils.config as configurations
import utils.dataloader as dataloader

config = configurations.local_config


def main():
    print("Hello World!")

    dataset = tf.data.Dataset.list_files(config["path"])\
        .map(dataloader.load_image)\
        .batch(config.batch_size)\
        .prefetch(2)


if __name__ == "__main__":
    main()
