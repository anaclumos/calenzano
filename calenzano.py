from Benchmark import Benchmark
from DataProvider import DataProvider
from models.resnet import ResNet, BasicBlock
from models.cnn import CNN
from models.mlp import MLP
import shutil


def main():

    shutil.rmtree("./dump", ignore_errors=True)

    models_config = [
        (ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10), "ResNet18"),
        (CNN(), "CNN"),
        (MLP(), "MLP")
    ]

    models = [model for model, _ in models_config]
    model_names = [name for _, name in models_config]

    transform_provider = DataProvider.get_instance("None")
    train_loader, test_loader = transform_provider.get_loaders()

    benchmark = Benchmark(models, model_names, train_loader, test_loader, epochs=10)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
