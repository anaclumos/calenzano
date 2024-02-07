import matplotlib.pyplot as plt

from Runner import Runner


class Benchmark:
    def __init__(self, models, model_names, trainloader, testloader, epochs=10):
        """
        Initialize the Benchmark class with models, their names, and data loaders.

        :param models: A list of model instances.
        :param model_names: A list of names corresponding to the models.
        :param trainloader: DataLoader for the training set.
        :param testloader: DataLoader for the test set.
        :param epochs: Number of epochs to train each model.
        """
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
            print(f"Running model: {name}")
            self.run_model(model, name)

        self.plot_results()

    def run_model(self, model, name):
        """
        Train and test a model, and save its accuracies.

        :param model: The model instance to run.
        :param name: The name of the model.
        """
        runner = Runner()
        runner.set_model(model, name)
        runner.train_and_test(self.trainloader, self.testloader, self.epochs)
        self.results[name] = runner.accuracies()

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
        plt.title("Model Comparison")
        plt.legend()
        plt.savefig("./dump/model_comparison.png")
        plt.show()
