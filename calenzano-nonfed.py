from Benchmark import Benchmark
from DataProvider import DataProvider
from models.cnn import CNN
from models.mlp import MLP
from models.resnet import BasicBlock, Bottleneck, ResNet


def main():

    models_config = [
        (ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10), "ResNet18"),
        (ResNet(BasicBlock, [3, 4, 6, 3], num_classes=10), "ResNet34"),
        (ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10), "ResNet50"),
        (ResNet(Bottleneck, [3, 4, 23, 3], num_classes=10), "ResNet101"),
        (ResNet(Bottleneck, [3, 8, 36, 3], num_classes=10), "ResNet152"),
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
