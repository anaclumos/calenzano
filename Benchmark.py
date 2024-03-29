import matplotlib.pyplot as plt

from Learner import Learner


class Benchmark:
    def __init__(
        self,
        models,
        model_names,
        trainloader,
        testloader,
        title="Model Comparison",
        epochs=10,
    ):
        """
        Initialize the Benchmark class with models, their names, and data loaders.

        :param models: A list of model instances.
        :param model_names: A list of names corresponding to the models.
        :param trainloader: DataLoader for the training set.
        :param testloader: DataLoader for the test set.
        :param epochs: Number of epochs to train each model.
        """
        self.title = title
        self.models = models
        self.model_names = model_names
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.results = {}

    def run_benchmark(self):
        """
        Run benchmarking for all models sequentially.
        """
        for model, name in zip(self.models, self.model_names):
            print(f"\n\n=== {name} ===\n")
            self.run_model(model, name)

        self.plot_results()

    def run_model(self, model, name):
        """
        Train and test a model, and save its accuracies.

        :param model: The model instance to run.
        :param name: The name of the model.
        """
        l = Learner(f"dump/{self.title}/{name}")
        l.set_model(model, name)
        l.train_and_test(self.trainloader, self.testloader, self.epochs)
        self.results[name] = l.accuracies()

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
