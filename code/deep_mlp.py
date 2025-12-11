import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import get_cifar10_loaders, SGD as CustomSGD, SGDMomentum as CustomSGDMomentum, Adam as CustomAdam

# set device and get loaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
trainloader, testloader = get_cifar10_loaders(batch_size=512, num_workers=8)


# deep MLP architecture
class DeepMLP(nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()
        # 4 layers as required with the activation layer
        self.fc1 = nn.Linear(32 * 32, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 10)
        self.activation = nn.Tanh()


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


# compute the misclassified fraction of images
def deep_mlp_compute_error(model, dataloader):
    model.eval()
    total = 0
    incorrect = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            incorrect += (predicted != labels).sum().item()

    return incorrect / total  # fraction of wrongly classified examples


# compute loss for a specific loader using the loss function
def deep_mlp_compute_loss(model, dataloader, loss_function):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)


# train the model
def deep_mlp_train_model(model, optimizer, loss_function, trainloader, testloader, num_epochs=20):
    model.train()
    losses = []
    train_errors = []
    test_losses = []
    test_errors = []

    for epoch in range(num_epochs):
        current_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()

        epoch_loss = current_loss / len(trainloader)
        losses.append(epoch_loss)

        # training and test errors/losses for this epoch
        train_error = deep_mlp_compute_error(model, trainloader)
        test_error = deep_mlp_compute_error(model, testloader)
        test_loss = deep_mlp_compute_loss(model, testloader, loss_function)

        train_errors.append(train_error)
        test_errors.append(test_error)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} | Train Error: {train_error:.4f} | Test Error: {test_error:.4f}")

    return losses, train_errors, test_losses, test_errors


# plot results
def deep_mlp_plot_results(best_results):
    # Training Loss
    plt.figure(figsize=(8,6))
    for name, res in best_results.items():
        plt.plot(res["losses"], label=f"{name} (lr={res['best_lr']})")
    plt.title("Deep MLP Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Test Loss
    plt.figure(figsize=(8,6))
    for name, res in best_results.items():
        plt.plot(res["test_losses"], label=f"{name} (lr={res['best_lr']})")
    plt.title("Deep MLP Test Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Training Error
    plt.figure(figsize=(8,6))
    for name, res in best_results.items():
        plt.plot(res["train_errors"], label=f"{name} (lr={res['best_lr']})")
    plt.title("Deep MLP Training Error per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Test Error
    plt.figure(figsize=(8,6))
    for name, res in best_results.items():
        plt.plot(res["test_errors"], label=f"{name} (lr={res['best_lr']})")
    plt.title("Deep MLP Test Error per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# run everything together for this architecture
def deep_mlp_run_experiments():
    loss_function = nn.CrossEntropyLoss()
    lr_options = {"SGD": [0.01, 0.05, 0.1], "SGD_momentum": [0.01, 0.05, 0.1], "Adam": [0.001, 0.005, 0.01]}
    best_results = {}

    for opt_name, lrs in lr_options.items():
        print(f"\n{opt_name}")
        best_acc = 0
        best_lr = None
        best_losses = []
        best_train_errors = []
        best_test_losses = []
        best_test_errors = []

        for lr in lrs:
            print(f"Currently using LR = {lr}")
            model = DeepMLP().to(device)

            # use custom optimizers defined
            if opt_name == "SGD":
                optimizer = CustomSGD(model.parameters(), lr=lr)
            elif opt_name == "SGD_momentum":
                optimizer = CustomSGDMomentum(model.parameters(), lr=lr, beta=0.1)
            elif opt_name == "Adam":
                optimizer = CustomAdam(model.parameters(), lr=lr, beta1=0.1, beta2=0.99)
            else:
                raise ValueError(f"Unknown optimizer: {opt_name}")

            start = time.time()
            losses, train_errors, test_losses, test_errors = deep_mlp_train_model(
                model, optimizer, loss_function, trainloader, testloader)
            acc = 100 * (1 - test_errors[-1])
            end = time.time()

            print(f"LR = {lr} | Test Accuracy: {acc:.2f}% | Time: {end - start:.2f}s")

            if acc > best_acc:
                best_acc = acc
                best_lr = lr
                best_losses = losses
                best_train_errors = train_errors
                best_test_losses = test_losses
                best_test_errors = test_errors

        print(f"\nBest LR for {opt_name} is {best_lr} (Acc={best_acc:.2f}%)")
        best_results[opt_name] = {
            "best_lr": best_lr,
            "best_acc": best_acc,
            "losses": best_losses,
            "train_errors": best_train_errors,
            "test_losses": best_test_losses,
            "test_errors": best_test_errors
        }

    # plot 4 separate figures
    deep_mlp_plot_results(best_results)

    return best_results


# run if script is executed
if __name__ == "__main__":
    print("Running Deep MLP Architecture")
    results = deep_mlp_run_experiments()
    for k, v in results.items():
        print(f"{k}: best_lr={v['best_lr']}, best_acc={v['best_acc']:.2f}%")
