# CANN (C Artificial Neural Network)

A lightweight neural network library implemented in pure C, designed for performance and educational purposes. This library provides a foundation for building and training neural networks from scratch without external dependencies.

## Features

- **Pure C Implementation**: Built from ground up in C without external dependencies
- **SIMD Optimization**: Uses ARM NEON SIMD instructions for optimized vector operations
- **Modular Architecture**: Clean separation of concerns with modular components
- **Multiple Activation Functions**:
  - ReLU
  - Sigmoid
  - Tanh
  - Softmax
- **Loss Functions**:
  - Mean Squared Error (MSE)
  - Cross-Entropy Binary (CEB)
- **Weight Initialization Methods**:
  - Xavier (Uniform & Normal)
  - He (Uniform & Normal)
- **Matrix Operations**: Matrix and vector operations

## Architecture

The library is organized into several key components:

- **Core Neural Network (`nn.c`, `nn.h`)**: Main neural network implementation
- **Matrix Operations (`matrix.c`, `matrix.h`)**: Matrix manipulation utilities
- **Vector Operations (`vector.c`, `vector.h`)**: Vector computation functions
- **Activation Functions (`activation.c`, `activation.h`)**: Various activation functions
- **Loss Functions (`loss.c`, `loss.h`)**: Loss function implementations
- **SIMD Optimizations (`simd_neon.c`, `simd_neon.h`)**: NEON SIMD accelerated operations
- **Random Distributions (`rand_distr.c`, `rand_distr.h`)**: Weight initialization utilities

## Building

The project uses a simple Makefile system. To build:

```bash
# Build the main library
make

# Clean build artifacts
make clean
```

### Requirements

- Clang compiler
- ARM processor with NEON support

## Usage

### Creating a Neural Network

```c
// Create a network with specified number of layers and learning rate
Network *net = create_network(3, 0.01);  // 3 layers, 0.01 learning rate

// Create and configure layers
Layer *l1 = create_layer(784, 128);  // Input -> Hidden
Layer *l2 = create_layer(128, 64);   // Hidden -> Hidden
Layer *l3 = create_layer(64, 10);    // Hidden -> Output

// Set activation functions
layer_set_activation(l1, make_activation_relu());
layer_set_activation(l2, make_activation_relu());
layer_set_activation(l3, make_activation_softmax());

// Initialize weights
layer_initialize(l1, uniform_xavier);
layer_initialize(l2, uniform_xavier);
layer_initialize(l3, uniform_xavier);

// Add layers to network
net_set_layer(net, l1, 0);
net_set_layer(net, l2, 1);
net_set_layer(net, l3, 2);

// Set loss function
net_set_loss(net, make_mse());
```

### Training

```c
// Train the network
net_train(net, X_train, Y_train, epochs);

// Make predictions
Vector *input = create_vector(784, true);
Vector *output = create_vector(10, true);
net_predict(net, input, output);
```

## Performance Optimizations

- SIMD acceleration using ARM NEON instructions
- Aligned memory allocation for better memory access patterns
- Efficient matrix and vector operations
- Batch processing support for training

## Memory Management

The library provides clear memory management functions:

- Creation functions (`create_*`)
- Destruction functions (`destroy_*`)
- Data management utilities for vectors and matrices

## License

This project is open source and available under the MIT License. 
