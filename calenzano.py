from torchvision.models import (alexnet, mobilenet_v2, resnet18, resnet34,
                                resnet50, resnet101, resnet152)

from Benchmark import Benchmark
from DataProvider import DataProvider


def main():

    models_config = [
        (resnet18(), "resnet18"),
        (resnet34(), "resnet34"),
        (resnet50(), "resnet50"),
        (resnet101(), "resnet101"),
        (resnet152(), "resnet152"),
        (alexnet(), "alexnet"),
        (mobilenet_v2(), "mobilenet_v2"),
    ]

    models = [model for model, _ in models_config]
    model_names = [name for _, name in models_config]

    transform_provider = DataProvider.get_instance("None")
    train_loader, test_loader = transform_provider.get_loaders()

    benchmark = Benchmark(models, model_names, train_loader, test_loader, epochs=10)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
