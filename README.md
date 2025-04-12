# CNN Implementation in C++

This project implements a Convolutional Neural Network (CNN) from scratch in C++, based on a modified version of LeNet-5 architecture for 28*28 images. The network is designed to recognize handwritten digits from the MNIST dataset.

## Architecture

The CNN consists of the following layers:
1. Input Layer (28x28 grayscale images)
2. Convolutional Layer 1 (6 filters of 5x5, stride 1)
3. Max Pooling Layer 1 (2x2 filter, stride 2)
4. Convolutional Layer 2 (16 filters of 5x5, stride 1)
5. Max Pooling Layer 2 (2x2 filter, stride 2)
6. Fully Connected Layer 1 (400 -> 120 neurons)
7. Fully Connected Layer 2 (120 -> 84 neurons)
8. Output Layer (84 -> 10 neurons)

## Technical details

We use min-batch (batch size 64) stochastic gradient descent for updating the weights with a learning rate of 0.001. We have not included any other form of optimizers as the implementation of these is very complicated and only adds towards computation time. We use ReLU as our activation function for the simplicity of calculating our gradients when passed through this layer, and also because the computation of other functions are more complicated and not suited for the given problem.

## Prerequisites

- C++17 compatible compiler
- GNU (version 11.4.0 or higher)
- OpenMP (version 4.5 or higher)
- CMake (version 3.22.1 or higher)
- WSL (version 2.5.6.0 or higher)

## Building the Project

Run the following commands on 

mkdir build
cd build
cmake ..
make -j$(nproc)

After all of these are completed, for running the CNN:

make run_cnn

And for testing the CNN:

make run_evaluate

And for running the simple neural network:

make -j$(nproc)

The program will:
1. Load the MNIST training and test data
2. Train the network for 5 epochs
3. Display the training loss after each epoch
4. Evaluate the model on the test set and display the accuracy

## Implementation Details

- The network uses ReLU activation functions for all layers except the output layer
- The output layer uses softmax activation
- Cross-entropy loss is used as the loss function
- The network is trained using stochastic gradient descent
- The implementation includes data normalization (pixel values scaled to 0-1)

## File details

- src: contains all our source files
   - cnn.cpp: implements the CNN
   - cnn_main.cpp: main function that runs the CNN
   - neural_network.cpp: implements the basic neural network
   - neural_network_main.cpp: main function that runs the basic neural network
   - evaluate.cpp: tests our CNN over test data and prints the confusion matrix
   - evaluate_main.cpp: main function that runs the evaluation
- include: contains the header files for cnn.cpp, evaluate.cpp and neural_network.cpp
- data: contains the source images and labels in unsigned byte format
- CMakeLists.txt: has the CMake file required for running the code
- cnn_output.txt: has the confusion matrix of the CNN model
- weights.bin: has the weights of the CNN network after training

## Results

The CNN gives us a testing accuracy of 95.16%, whereas our simple neural network gives us a testing accuracy of 72.10%, showing how the spacial awareness of a CNN benefits in having a better model. The confusion matrix produced by the CNN model is present in the file cnn_output.txt

## Additional Notes

- We have used multiple compiler optimizations and OpenMP to optimize our CNN traininf code, because training time was extremely high without these optimizations.
- The logic and formulae used for implementing forwarding and backpropogation in the convolution layers in our neural network was taken from the following YouTube video: https://www.youtube.com/watch?v=z9hJzduHToc
- Our basic neural network was made solely for comparing the results to our CNN with a basic neural network approach, hence we have not made the evaluation files for the basic neural network.
- To modify the hyperparameters, go to CMakeLists.txt and modify the numbers mentioned alongside the comments