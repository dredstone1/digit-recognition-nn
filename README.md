# 🔢 Digit Recognition Neural Network

A C++ application for handwritten digit recognition that leverages the [NeuralNetwork](https://github.com/dredstone1/NeuralNetwork) library. This project demonstrates how to use a modern neural network implementation to recognize digits (0-9) from images with high accuracy.


## ✨ Key Features

- **📊 MNIST Dataset Integration**: Ready-to-use solution for the standard benchmark in digit recognition
- **🎮 User-friendly Interface**: Simple commands to train and test the neural network
- **📈 Performance Visualization**: Monitoring of accuracy and loss during training
- **🖼️ Prediction Visualization**: See the network's predictions on digit images
- **⚙️ Configuration Options**: Customize network architecture and training parameters
- **💾 Model Saving/Loading**: Preserve trained models for later use

## 🔗 Dependencies

- **[NeuralNetwork Library](https://github.com/dredstone1/NeuralNetwork)**: The core neural network implementation providing:
  - Modern C++17 implementation
  - Flexible network architecture
  - Various activation functions
  - Efficient training algorithms
  - Model persistence capabilities

## 🏗️ Quick Start

### Prerequisites

- 🛠️ C++17-compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- 📋 CMake 3.28 or later
- 📚 NeuralNetwork library (included as a submodule)

### 🔨 Build Instructions

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/dredstone1/digit-recognition-nn.git

# Navigate to project directory
cd digit-recognition-nn

# Create and enter build directory
mkdir build && cd build

# Configure and build
cmake ..
make
```

### 🏃‍♂️ Running the Program

```bash
# Run the digit recognition program with loading a pre-trained model
./digit_recognition -l

# Train a new model
./digit_recognition -t

# Continue training the model
./digit_recognition -l -t
```

## 📊 Performance

The application achieves approximately 90-93% accuracy on the MNIST test dataset after training, making it suitable for many practical applications requiring digit recognition.

## 🧩 How It Works

This application:

1. **Loads the MNIST dataset** into memory
2. **Creates a neural network** using the NeuralNetwork library with an architecture optimized for digit recognition
3. **Trains the network** on the training set (or loads a pre-trained model)
4. **Evaluates the network** on the test set to measure accuracy
5. **Allows interactive testing** with user-provided digit images

## 🛠️ Command Line Options

| Flag | Description |
|------|-------------|
| `-t` | Train a new model |
| `-l` | Load a pre-trained model |

## 🤝 Contributing

Contributions are welcome! Whether you want to improve accuracy, optimize performance, or add new features:

- 🐛 Open issues for bugs or feature requests
- 🔀 Submit pull requests with your improvements
- 💡 Share ideas and suggestions
- 📖 Help improve documentation

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete details.

## 👨‍💻 Author

**dredstone1** - [GitHub Profile](https://github.com/dredstone1)

---

⭐ If you find this project helpful for your digit recognition tasks, consider giving it a star!
