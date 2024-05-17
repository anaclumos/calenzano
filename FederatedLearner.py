import copy
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class FederatedLearner:
    def __init__(self, base_model, base_path="./dump", strategy="AVG"):
        self.base_model = base_model
        if torch.cuda.is_available():
            print(f"Using NVIDIA GPU")
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print(f"Using Metal Acceleration (MPS) on Apple GPU")
            self.device = torch.device("mps")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")
        self.global_model = copy.deepcopy(self.base_model).to(self.device)
        self.base_path = base_path
        self.model_path = os.path.join(self.base_path, "models")
        self.accuracy_path = os.path.join(self.base_path, "accuracies.json")
        self.accuracy_plot_path = os.path.join(self.base_path, "accuracy_plot.png")
        os.makedirs(self.model_path, exist_ok=True)
        self.accuracies = []
        self.strategy = strategy

    def train_local_models(self, loaders_list, global_model, epochs=1):
        """
        Train local models for one epoch and return them.
        """
        local_models = []
        criterion = nn.CrossEntropyLoss()
        id = 0
        for trainloader in loaders_list:
            id += 1
            print(f"Training local model {id}/{len(loaders_list)}")
            model = copy.deepcopy(global_model).to(self.device)  # Start from the current global model
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

            model.train()
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            local_models.append(model)
        return local_models

    def aggregate_models(self, local_models):
        """
        Aggregate the models by averaging their weights and update the global model.
        Only aggregates parameters of floating point type.
        """
        global_dict = self.global_model.state_dict()

        if self.strategy == "AVG":
            for key in global_dict.keys():
                if "float" in str(global_dict[key].dtype):  # Check if the parameter is of floating point type
                    # Aggregate only if the parameter is of floating point type
                    global_dict[key] = torch.mean(
                        torch.stack([model.state_dict()[key].float().cpu() for model in local_models]),
                        0
                    ).to(self.device)

        elif self.strategy == "AVG_NONZERO":
            for key in global_dict.keys():
                if "float" in str(global_dict[key].dtype):
                    # Aggregate only if the parameter is of floating point type
                    non_zero_params = [
                        model.state_dict()[key].float().cpu()
                        for model in local_models
                        if not torch.all(model.state_dict()[key].float() == 0)
                    ]
                    if len(non_zero_params) > 0:
                        global_dict[key] = torch.mean(torch.stack(non_zero_params), 0).to(self.device)

        elif self.strategy == "MAX":
            for key in global_dict.keys():
                if "float" in str(global_dict[key].dtype):
                    # Aggregate only if the parameter is of floating point type
                    global_dict[key] = torch.max(
                        torch.stack([model.state_dict()[key].float().cpu() for model in local_models]),
                        0
                    ).values.to(self.device)

        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

        self.global_model.load_state_dict(global_dict)

    def test_global_model(self, testloader):
        correct = 0
        total = 0
        self.global_model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def federated_train(self, loaders_list, testloader, epochs=10):
        for epoch in range(epochs):
            local_models = self.train_local_models(loaders_list, self.global_model, epochs=1)
            self.aggregate_models(local_models)
            accuracy = self.test_global_model(testloader)
            self.accuracies.append(accuracy)
            self.save_model(epoch + 1, accuracy)
            self.update_accuracy_json(self.accuracies)
            print(f"Epoch {epoch+1}/{epochs}, Global Model Accuracy: {accuracy:.2f}%")

        self.plot_accuracies(self.accuracies)
        print("Finished Federated Training")

    def save_model(self, epoch, accuracy):
        filename = f"{self.model_path}/epoch_{str(epoch).zfill(3)}_accuracy_{int(accuracy * 100)}.pth"
        torch.save(self.global_model.state_dict(), filename)

    def update_accuracy_json(self, accuracies):
        with open(self.accuracy_path, "w") as f:
            json.dump(accuracies, f)

    def plot_accuracies(self, accuracies):
        epochs = list(range(1, len(accuracies) + 1))
        plt.plot(epochs, accuracies, "-o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy of Global Model")
        plt.savefig(self.accuracy_plot_path, dpi=1200)
        plt.clf()
