from DataProvider import DataProvider
from FederatedBenchmark import FederatedBenchmark
from models.mlp import QuadLayerFC_FlattenReLU_MLP
from TransformProvider import ScatterTransform
from utils import plot_transformed_images


def main():

    models_config = [
        (QuadLayerFC_FlattenReLU_MLP(num_classes=10), "AVG_MLP", "AVG"),
        (QuadLayerFC_FlattenReLU_MLP(num_classes=10), "AVG_NONZERO_MLP", "AVG_NONZERO"),
        (QuadLayerFC_FlattenReLU_MLP(num_classes=10), "MAX_MLP", "MAX"),
    ]

    models = [model for model, _, _ in models_config]
    model_names = [name for _, name, _ in models_config]
    federation_strategy = [strategy for _, _, strategy in models_config]

    loaders = []
    total = 4
    title = f"Scatter{total}Train FullTest 3MLP"

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
        epochs=1000,
        strategies=federation_strategy,
    )

    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
