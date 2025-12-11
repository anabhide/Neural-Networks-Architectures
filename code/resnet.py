import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from utils import get_cifar10_loaders, custom_glogot_normal, SGD as CustomSGD, SGDMomentum as CustomSGDMomentum, Adam as CustomAdam

# get training and test loaders using our custom CIFAR-10 grayscale loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
trainloader, testloader = get_cifar10_loaders(batch_size=512, num_workers=8)


# Residual Network (ResNet) architecture
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # Layer 1 & 2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.proj1 = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.proj2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(64 * 8 * 8, 10)

        # Glorot / Xavier initialization for the layers
        for layer in [self.conv1, self.conv2, self.proj1, self.conv3, self.conv4, self.proj2, self.fc]:
            custom_glogot_normal(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


    def forward(self, x):
        # 1st residual block
        out = self.relu(self.conv1(x))
        out2 = self.conv2(out)
        proj_x = self.proj1(x)
        res1 = self.relu(out2 + proj_x)
        pooled = self.pool1(res1)

        # 2nd residual block
        out = self.relu(self.conv3(pooled))
        out2 = self.conv4(out)
        proj_pooled = self.proj2(pooled)
        res2 = self.relu(out2 + proj_pooled)
        out = self.pool2(res2)

        out = self.flatten(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out



# calculate fraction of incorrect predictions
def resnet_compute_error(model, dataloader):
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
def resnet_compute_loss(model, dataloader, loss_function):
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
def resnet_train_model(model, optimizer, loss_function, trainloader, testloader, num_epochs=20):
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
        train_error = resnet_compute_error(model, trainloader)
        test_error = resnet_compute_error(model, testloader)
        test_loss = resnet_compute_loss(model, testloader, loss_function)

        train_errors.append(train_error)
        test_errors.append(test_error)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} | Train Error: {train_error:.4f} | Test Error: {test_error:.4f}")

    return losses, train_errors, test_losses, test_errors


# plot figures
def resnet_plot_results(best_results):
    # Training Loss
    plt.figure(figsize=(8,6))
    for name, res in best_results.items():
        plt.plot(res["losses"], label=f"{name} (lr={res['best_lr']})")
    plt.title("Training Loss per Epoch for ResNet")
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
    plt.title("Test Loss per Epoch for ResNet")
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
    plt.title("Training Error per Epoch for ResNet")
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
    plt.title("Test Error per Epoch for ResNet")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# run experiment for this architecture
def resnet_run_experiments():
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
            model = ResNet().to(device)

            if opt_name == "SGD":
                optimizer = CustomSGD(model.parameters(), lr=lr)
            elif opt_name == "SGD_momentum":
                optimizer = CustomSGDMomentum(model.parameters(), lr=lr, beta=0.1)
            elif opt_name == "Adam":
                optimizer = CustomAdam(model.parameters(), lr=lr, beta1=0.1, beta2=0.99)

            start = time.time()
            # train (now tracks test metrics each epoch)
            losses, train_errors, test_losses, test_errors = resnet_train_model(
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

    resnet_plot_results(best_results)

    return best_results


# if script run directly
if __name__ == "__main__":
    print("Running Residual Networks Architecture")
    results = resnet_run_experiments()
    for k, v in results.items():
        print(f"{k}: best_lr={v['best_lr']}, best_acc={v['best_acc']:.2f}%")
