# Neural Network Architectures & Optimization from Scratch

Implements and compares 8 neural network architectures on CIFAR-10, built with low-level PyTorch
(manual layers and training loops, not high-level `nn.Sequential` wrappers) to understand what each
architectural choice actually contributes to performance.

## Architectures implemented
- Single-layer perceptron (linear classifier)
- Shallow MLP
- Deep MLP
- Deep MLP with ReLU
- CNN
- CNN with dropout
- Deep CNN (VGG-style)
- Residual Network (ResNet)

## Optimizers
Three optimizers: SGD, Momentum, and Adam were implemented and compared across architectures to study
how optimizer choice interacts with model depth and capacity.

## What this shows
- How accuracy scales from a linear classifier up through deep residual networks (best: **86% test
  accuracy** on CIFAR-10)
- How dropout affects generalization in CNNs
- How residual connections change training stability in deeper networks
- Effect of Glorot/Xavier initialization on convergence

Each architecture has its own training script under `code/`, with accuracy/loss curves saved under `images/`.

## Stack
PyTorch, NumPy
