# MNIST C

## Overview

Train a CNN neural network purely in C on the MNIST dataset.

## Models

- Multi-layer perceptron
- CNN

## Process

- Load dataset
- Preprocess data
- Dataloader
- Train for N epochs
  - Load batch
  - Forward pass
  - Backward pass
- Classification model?

## TODO

- [x] Train MNIST in Python
   - [x] Torch (First)
   - [x] Implement addition task to learn gradient descent
   - [x] Implement basic linear classifier (OR gate)
   - [ ] Implement basic MLP classifier (XOR gate)

- [ ] Train MNIST in C (Port python)
   - Manual matrix operations or library?
   <!-- - Add Metal support for MacOS? -->