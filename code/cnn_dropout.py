import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from utils import get_cifar10_loaders, custom_glogot_normal, SGD as CustomSGD, SGDMomentum as CustomSGDMomentum, Adam as CustomAdam

# get training and test loaders using our custom CIFAR-10 grayscale loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
trainloader, testloader = get_cifar10_loaders(batch_size=512, num_workers=8)


# CNN with Dropout architecture
class CNN_Dropout(nn.Module):
    def __init__(self):
        super(CNN_Dropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(64 * 8 * 8, 10)

        # Xavier (Glorot) initialization for conv + fc
        custom_glogot_normal(self.conv1.weight)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        custom_glogot_normal(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        custom_glogot_normal(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x



# calculate fraction of incorrect predictions
def cnn_dropout_compute_error(model, dataloader):
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


# calculate loss over a dataloader (train/test)
def cnn_dropout_compute_loss(model, dataloader, loss_function):
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
def cnn_dropout_train_model(model, optimizer, loss_function, trainloader, testloader, num_epochs=20):
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

        train_error = cnn_dropout_compute_error(model, trainloader)
        test_error = cnn_dropout_compute_error(model, testloader)
        test_loss = cnn_dropout_compute_loss(model, testloader, loss_function)

        train_errors.append(train_error)
        test_errors.append(test_error)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} | Train Error: {train_error:.4f} | Test Error: {test_error:.4f}")

    return losses, train_errors, test_losses, test_errors


# plot figures
def cnn_dropout_plot_results(best_results):
    plt.figure(figsize=(8,6))
    for name, res in best_results.items():
        plt.plot(res["losses"], label=f"{name} (lr={res['best_lr']})")
    plt.title("Training Loss per Epoch for CNN with Dropout")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,6))
    for name, res in best_results.items():
        plt.plot(res["test_losses"], label=f"{name} (lr={res['best_lr']})")
    plt.title("Test Loss per Epoch for CNN with Dropout")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,6))
    for name, res in best_results.items():
        plt.plot(res["train_errors"], label=f"{name} (lr={res['best_lr']})")
    plt.title("Training Error per Epoch for CNN with Dropout")
    plt.xlabel("Epoch"); plt.ylabel("Error")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,6))
    for name, res in best_results.items():
        plt.plot(res["test_errors"], label=f"{name} (lr={res['best_lr']})")
    plt.title("Test Error per Epoch for CNN with Dropout")
    plt.xlabel("Epoch"); plt.ylabel("Error")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


# run experiment for this architecture
def cnn_dropout_run_experiments():
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
            model = CNN_Dropout().to(device)

            if opt_name == "SGD":
                optimizer = CustomSGD(model.parameters(), lr=lr)
            elif opt_name == "SGD_momentum":
                optimizer = CustomSGDMomentum(model.parameters(), lr=lr, beta=0.1)
            elif opt_name == "Adam":
                optimizer = CustomAdam(model.parameters(), lr=lr, beta1=0.1, beta2=0.99)

            start = time.time()
            losses, train_errors, test_losses, test_errors = cnn_dropout_train_model(
                model, optimizer, loss_function, trainloader, testloader
            )

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

    cnn_dropout_plot_results(best_results)
    return best_results


if __name__ == "__main__":
    print("Running CNN with Dropout (p=0.5) Architecture")
    results = cnn_dropout_run_experiments()
    for k, v in results.items():
        print(f"{k}: best_lr={v['best_lr']}, best_acc={v['best_acc']:.2f}%")
