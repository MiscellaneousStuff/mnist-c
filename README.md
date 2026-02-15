# MNIST C

## Overview

Train an MLP neural network purely in C on the MNIST dataset.

## Models

- [ ] Multi-layer perceptron

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
   - [x] Implement basic MLP classifier (XOR gate)
   - [x] Replace torch MLP with manual MLP and train MNIST

- [ ] Train MNIST in C (Port python)
   - [ ] Port XOR MLP task
      - [ ] Forward pass
      - [ ] Backward pass
   - [ ] Port MNIST MLP task
      - [ ] Port dataloader
      - [ ] Compare accuracy vs python