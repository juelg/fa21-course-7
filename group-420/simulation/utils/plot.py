import matplotlib.pyplot as plt


def plot(image):
    plt.imshow((image.numpy() * 50).astype("uint8"))
    plt.show()
