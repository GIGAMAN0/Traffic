# Traffic Sign Recognition with Neural Networks

This project aims to develop an AI system to identify traffic signs from photographs using neural networks, particularly with TensorFlow.

## Introduction

Traffic sign recognition is a crucial component of autonomous driving systems, enabling vehicles to understand and interpret their surroundings from images captured by cameras. In this project, we utilize TensorFlow to build a neural network model that can classify different types of road signs based on input images.

## Files

- `traffic.py`: Contains the implementation of functions to load data and build the neural network model.
- `README.md`: Documentation of experimentation process and observations.
- `requirements.txt`: List of required Python packages.

## Usage

1. Download the dataset from the provided link and unzip it.
2. Move the dataset directory inside the `traffic` directory.
3. Install dependencies by running `pip3 install -r requirements.txt`.
4. Run the command `python traffic.py <path_to_dataset>` to train and evaluate the model.

## Experimentation Process

For the experimentation process, I focused on tuning the architecture of the neural network model to achieve better accuracy in classifying traffic signs. Here's a summary of my experimentation:

- **Convolutional Layers**: I experimented with different numbers and sizes of convolutional layers to capture spatial features from input images effectively.
- **Pooling Layers**: Various pool sizes for pooling layers were tested to downsample the feature maps while retaining important information.
- **Hidden Layers**: I tried different numbers of hidden layers and neurons to find the optimal balance between model complexity and performance.
- **Dropout**: Dropout regularization was applied to prevent overfitting by randomly dropping neurons during training.
- **Activation Functions**: I explored different activation functions such as ReLU and tanh to introduce non-linearity into the model.
- **Learning Rate**: Adjustments to the learning rate were made to control the rate at which the model learns from the training data.

## Observations

- Increasing the depth of the neural network by adding more layers led to better performance on the training set but also increased the risk of overfitting on the test set.
- Dropout regularization helped mitigate overfitting and improved the generalization ability of the model.
- The choice of activation function significantly influenced the convergence speed and final accuracy of the model.
- Experimenting with different hyperparameters requires careful tuning and experimentation to find the optimal configuration for the specific task.
