import matplotlib.pyplot as plt
import numpy as np

from DataProvider import DataProvider
from FederatedBenchmark import FederatedBenchmark
from models.cnn import CNN
from models.mlp import MLP
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from TransformProvider import ScatterTransform
from utils import plot_transformed_images


def main():

    models_config = [
        (MLP(num_classes=10), "MLP"),
        (CNN(num_classes=10), "CNN"),
        (resnet18(num_classes=10), "ResNet18"),
        (resnet34(num_classes=10), "ResNet34"),
    ]

    models = [model for model, _ in models_config]
    model_names = [name for _, name in models_config]

    loaders = []
    total = 4
    title = f"Scatter{total}Train FullTest Benchmark"

    for i in range(total):
        key = f"scatter_{i}"
        transform = ScatterTransform(seed=42, idx=i, desired_dataset_count=total)
        provider = DataProvider.get_instance(key, transform)
        trainloader, _ = provider.get_loaders()
        loaders.append(trainloader)

    _, full_testloader = DataProvider.get_instance(None).get_loaders()

    plot_transformed_images(trainloader, save_path=f"./dump/{title}/example.png")

    benchmark = FederatedBenchmark(
        models,
        model_names,
        loaders,
        full_testloader,
        title=title,
        epochs=50,
    )

    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
