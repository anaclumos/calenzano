import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


class Runner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_name = ""
        self.init_paths()

    def init_paths(self, model_name=""):
        base_path = "./dump"
        if model_name:
            self.model_path = os.path.join(base_path, model_name, "models")
            self.accuracy_path = os.path.join(base_path, model_name, "accuracies.json")
            self.accuracy_plot_path = os.path.join(
                base_path, model_name, "accuracy_plot.png"
            )
            os.makedirs(self.model_path, exist_ok=True)

    def set_model(self, model, model_name):
        self.model = model.to(self.device)
        self.model_name = model_name
        self.init_paths(model_name)

    def save_model(self, epoch, accuracy):
        filename = f"{self.model_path}/epoch_{str(epoch).zfill(3)}_accuracy_{int(accuracy*100)}.pth"
        torch.save(self.model.state_dict(), filename)

    def update_accuracy_json(self, accuracies):
        with open(self.accuracy_path, "w") as f:
            json.dump(accuracies, f)

    def plot_accuracies(self, accuracies):
        epochs = list(range(1, len(accuracies) + 1))
        plt.plot(epochs, accuracies, "-o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Improvement Over Epochs")
        plt.savefig(self.accuracy_plot_path, dpi=1200)

    def accuracies(self):
        with open(self.accuracy_path, "r") as f:
            accuracies = json.load(f)
        return accuracies

    def train_and_test(self, trainloader, testloader, epochs=10):
        if not self.model:
            raise ValueError(
                "Model not set. Use set_model() to set a model before training."
            )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        accuracies = []

        for epoch in range(epochs):
            self.model.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            accuracy = self.test_net(testloader)
            accuracies.append(accuracy)
            self.save_model(epoch + 1, accuracy)
            self.update_accuracy_json(accuracies)
            self.plot_accuracies(accuracies)
            print(f"Epoch {epoch+1}, Accuracy: {accuracy:.2f}%")

        print("Finished Training and Testing")

    def test_net(self, testloader):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy
