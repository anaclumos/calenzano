import matplotlib.pyplot as plt
import numpy as np
import os


def imshow(img):
    """
    Function to show an image.
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_transformed_images(loader, num_images=4, save_path=None):
    """
    Function to plot example images to verify the transformations.
    """
    dataiter = iter(loader)
    images, labels = next(dataiter)

    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        imshow(images[i])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=1200)
    plt.show()
