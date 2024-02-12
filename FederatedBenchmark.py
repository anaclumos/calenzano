import matplotlib.pyplot as plt

from FederatedLearner import FederatedLearner


class FederatedBenchmark:
    def __init__(
        self,
        models,
        model_names,
        loaders,
        testloader,
        title="Model Comparison",
        epochs=10,
    ):
        self.title = title
        self.models = models
        self.model_names = model_names
        self.loaders = loaders  # Assuming this is a list of training loaders for federated training
        self.testloader = testloader  # A single test loader for evaluating the global model
        self.epochs = epochs
        self.results = {}  # Initialize the results dictionary here

    def run_benchmark(self):
        """
        Run benchmarking for all models sequentially, accommodating the federated learning setup.
        """
        for model, name in zip(self.models, self.model_names):
            print(f"\n\n=== {name} ===\n")
            self.run_model(model, name)

        self.plot_results()

    def run_model(self, model, name):
        federated_learner = FederatedLearner(model, f'dump/{self.title}/{name}')
        federated_learner.federated_train(self.loaders, self.testloader, self.epochs)
        self.results[name] = federated_learner.accuracies  # Store accuracies for each model

    def plot_results(self):
        """
        Plot the accuracies of all models on the same plot for comparison.
        """
        plt.figure(figsize=(10, 6))
        for name, accuracies in self.results.items():
            epochs = list(range(1, len(accuracies) + 1))
            plt.plot(epochs, accuracies, "-o", label=name)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(self.title)
        plt.legend()
        plt.savefig(f"./dump/{self.title}/accuracy.png", dpi=1200)
        plt.clf()
