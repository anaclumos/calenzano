from Benchmark import Benchmark
from DataProvider import DataProvider
from models.cnn import CNN
from models.mlp import MLP
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


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

    transform_provider = DataProvider.get_instance("None")
    train_loader, test_loader = transform_provider.get_loaders()

    benchmark = Benchmark(
        models,
        model_names,
        train_loader,
        test_loader,
        epochs=50,
        title="FullTrain FullTest Benchmark",
    )
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
