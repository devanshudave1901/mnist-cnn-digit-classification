This project was developed as part of **SFWRTECH/ENGTECH 4AI3 – Artificial Intelligence** at **McMaster University**. The assignment focused on understanding and implementing a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset.

The assignment consisted of two main components:

**Question 1 – CNN Architecture and Parameter Calculation:** Design a Python program to calculate the number of trainable parameters (weights and biases) for a specified CNN architecture. This involved determining the output dimensions of the convolutional and max pooling layers, calculating the flattened input size, and computing the parameters of the convolutional, hidden, and output layers.

**Question 2 – CNN Implementation and Evaluation:** Adapt the instructor-provided `Simple_CNN_for_the_MNIST_Dataset.ipynb` notebook to match the required architecture, including the specified convolutional filters, padding, stride, pooling configuration, and fully connected layers. The updated model was then trained and evaluated on the MNIST dataset to measure its classification accuracy.

The final implementation combines **mathematical parameter calculations with practical deep learning using Python and TensorFlow/Keras**, providing hands-on experience with CNN architecture, training, and evaluation.

# Handwritten Digit Classification Using CNN

A deep learning project that uses a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) from the MNIST dataset. Built with **Python and TensorFlow/Keras**, this project demonstrates image classification, feature extraction, neural network architecture, parameter calculations, and model evaluation.

## Project Overview

The objective is to develop a CNN capable of identifying handwritten digits from grayscale images.

The network processes input images through convolutional and pooling layers to extract meaningful visual features, followed by fully connected layers to classify each image into one of ten digit categories.

The project also includes mathematical calculations to determine the number of trainable parameters in each layer and the overall neural network.

## Dataset — MNIST

The **MNIST (Modified National Institute of Standards and Technology)** dataset is a widely used benchmark for handwritten digit recognition.

| Property | Description |
|---|---|
| Dataset | MNIST |
| Training images | 60,000 |
| Testing images | 10,000 |
| Image dimensions | 28 × 28 pixels |
| Image channels | 1 (Grayscale) |
| Number of classes | 10 (Digits 0–9) |
| Task | Multi-class classification |

Each image contains 784 pixels representing grayscale intensity values.

### Data Preprocessing

Before training, the images and labels are prepared through the following steps:

1. **Reshaping:** Images are converted to the shape `(28, 28, 1)` to include the channel dimension.
2. **Type conversion:** Pixel values are converted to floating-point numbers.
3. **Normalization:** Pixel values are scaled from the range 0–255 to 0–1.
4. **One-hot encoding:** Class labels are transformed into vectors representing the ten digit categories.

These steps prepare the dataset for training and evaluation.

## CNN Architecture

The model consists of one convolutional layer, one max pooling layer, a flatten layer, one fully connected hidden layer, and an output layer.

**Architecture flow:**

Input Image (28 × 28 × 1)

↓

Zero Padding (1 pixel on each side)

↓

Convolutional Layer — 70 Filters (5 × 5), Stride 2, ReLU

↓

Max Pooling Layer — Pool Size (2 × 2), Stride 1

↓

Flatten Layer

↓

Fully Connected Hidden Layer — 32 Neurons, ReLU

↓

Output Layer — 10 Neurons, Softmax

↓

**Predicted Digit (0–9)**

The implementation is based on the Sequential model provided by Keras.

### Layer Configuration

| Layer | Configuration | Output Shape |
|---|---|---|
| Input | Grayscale image | 28 × 28 × 1 |
| Zero Padding | Padding = 1 | 30 × 30 × 1 |
| Convolution | 70 filters, 5×5, stride 2 | 13 × 13 × 70 |
| Max Pooling | 2×2, stride 1 | 12 × 12 × 70 |
| Flatten | Converts feature maps to 1D | 10,080 |
| Dense Hidden | 32 neurons, ReLU | 32 |
| Dense Output | 10 neurons, Softmax | 10 |

The starter implementation also includes an optional Dropout layer with a rate of 0.2 to reduce overfitting during training.

## Trainable Parameter Calculations

An important part of this project involves calculating the total number of trainable parameters in the neural network.

Trainable parameters consist of **weights and biases** that are updated during model training.

### 1. Convolutional Layer

The number of trainable parameters is calculated using:

**Parameters = (Kernel Height × Kernel Width × Input Channels + 1) × Number of Filters**

For this architecture:

(5 × 5 × 1 + 1) × 70

**Convolutional Parameters = 1,820**

The additional 1 accounts for the bias associated with each filter.

### 2. Max Pooling and Flatten Layers

Max pooling selects the maximum value within a pooling region, while flattening converts multidimensional feature maps into a one-dimensional vector.

Neither operation introduces trainable weights or biases.

**Max Pooling Parameters = 0**

**Flatten Parameters = 0**

### 3. Fully Connected Hidden Layer

After max pooling, the output dimensions are:

12 × 12 × 70 = 10,080 inputs

The hidden layer contains 32 neurons.

**Dense Parameters = (Input Features + 1) × Number of Neurons**

(10,080 + 1) × 32

**Hidden Layer Parameters = 322,592**

### 4. Output Layer

The output layer consists of 10 neurons representing the ten possible digit classes.

(32 + 1) × 10

**Output Layer Parameters = 330**

### Total Trainable Parameters

| Layer | Parameters |
|---|---:|
| Convolutional Layer | 1,820 |
| Max Pooling | 0 |
| Flatten | 0 |
| Dense Hidden Layer | 322,592 |
| Dense Output Layer | 330 |
| **Total** | **324,742** |

The calculation is performed programmatically using Python and can be verified using Keras's `model.summary()`.

## Model Training

The model is trained using the MNIST training dataset.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Framework | TensorFlow/Keras |
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Evaluation Metric | Accuracy |
| Epochs | 10 |
| Batch Size | 200 |
| Output Activation | Softmax |

The Adam optimizer updates the model's weights during training to minimize classification error.

Categorical cross-entropy measures the difference between predicted class probabilities and the actual encoded labels.

The Softmax activation function produces a probability distribution across the ten output classes.

## Model Evaluation

After training, the model is evaluated on the MNIST test dataset containing 10,000 handwritten digit images.

The primary evaluation metric is **classification accuracy**, calculated as:

**Accuracy = (Correct Predictions / Total Predictions) × 100**

Classification error is calculated as:

**Error Rate = 100% − Accuracy**

### Results

| Metric | Result |
|---|---|
| Total Trainable Parameters | 324,742 |
| Training Epochs | 10 |
| Test Dataset Size | 10,000 |
| Final Test Accuracy | Add verified result from the final 70-filter model |

The notebook also demonstrates predictions on individual handwritten digits, allowing the model's predicted class and actual label to be compared.

*Note: Accuracy may vary between training runs because of initialization and other training conditions.*

## Technologies Used

- **Python** — Programming language
- **TensorFlow/Keras** — Deep learning framework
- **NumPy** — Numerical computation and data processing
- **Matplotlib** — Image visualization
- **Jupyter Notebook** — Interactive development and execution

## Getting Started

### Prerequisites

Install Python 3 and Jupyter Notebook.

Install the
