import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from DataProvider import DataProvider
from FederatedBenchmark import FederatedBenchmark
from models.cnn import CNN
from models.mlp import MLP
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from TransformProvider import GridTransform


def imshow(img):
    """
    Function to show an image.
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_transformed_images(loader, num_images=4):
    """
    Function to plot example images to verify the transformations.
    """
    dataiter = iter(loader)
    images, labels = next(dataiter)

    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        imshow(images[i])


def main():

    models_config = [
        (resnet18(num_classes=10), "ResNet18"),
        (resnet34(num_classes=10), "ResNet34"),
        (resnet50(num_classes=10), "ResNet50"),
        (resnet101(num_classes=10), "ResNet101"),
        (resnet152(num_classes=10), "ResNet152"),
        (CNN(num_classes=10), "CNN"),
        (MLP(num_classes=10), "MLP"),
    ]

    models = [model for model, _ in models_config]
    model_names = [name for _, name in models_config]

    loaders = []

    for row in range(2):
        for col in range(2):
            key = f"row{row}_col{col}"
            transform = GridTransform(row, col)
            provider = DataProvider.get_instance(key, transform)
            trainloader, testloader = provider.get_loaders()
            loaders.append(trainloader)

    _, full_testloader = DataProvider.get_instance(None).get_loaders()

    benchmark = FederatedBenchmark(
        models,
        model_names,
        loaders,
        full_testloader,
        title="2x2Train FullTest Benchmark",
        epochs=50,
    )

    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
