from DataProvider import DataProvider
from FederatedBenchmark import FederatedBenchmark
from models.cnn import DualConvPair_MaxPool_DualFCReLU_CNN, TripleConvPoolReLU_FlatFC_CNN
from models.mlp import QuadLayerFC_FlattenReLU_MLP
from TransformProvider import ScatterAllowingDupedTransform
from utils import plot_transformed_images


def main():

    models_config = [
        (TripleConvPoolReLU_FlatFC_CNN(num_classes=10), "TripleConvPoolReLU_FlatFC_CNN"),
        (DualConvPair_MaxPool_DualFCReLU_CNN(num_classes=10), "DualConvPair_MaxPool_DualFCReLU_CNN"),
        (QuadLayerFC_FlattenReLU_MLP(num_classes=10), "QuadLayerFC_FlattenReLU_MLP"),
    ]

    models = [model for model, _ in models_config]
    model_names = [name for _, name in models_config]

    loaders = []
    total = 4
    title = f"Scatter{total}AllowingDupedTrain FullTest Max-AvgNonZero Benchmark (CNN-MLP 200 Epoches)"

    for i in range(total):
        key = f"scatter_{i}"
        transform = ScatterAllowingDupedTransform(idx=i, desired_dataset_count=total)
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
        epochs=200,
    )

    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
