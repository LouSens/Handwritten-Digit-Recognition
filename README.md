# MNIST Digit Recognition Project

## Overview
This project is to develop a digit recognition system using a convolutional neural network (CNN) trained on the MNIST dataset. The program includes an intuitive graphical user interface (GUI) that allows users to draw digits and receive accurate predictions. This project highlights the application of deep learning for solving real-world image recognition problems.


## Features
1. **Comprehensive Functionality**: The program meets all specified requirements, offering a seamless user experience for digit recognition.
2. **Enhanced Model Architecture**: Utilizes TensorFlow to design and train a CNN with two convolutional blocks, dropout layers, and dense layers for optimal performance.
3. **Interactive GUI**: Built using Tkinter, providing users with an easy-to-use canvas for drawing digits and viewing predictions.
4. **Data Augmentation**: Improves generalization through transformations like rotation, zoom, shifts, and shearing.
5. **Code Quality**: Organized with clear structure, reusable methods, and meaningful variable names, following best practices in Python development.
6. **Individual Work and Contributions**: Every element of this project reflects my personal effort, commitment, and expertise in delivering a robust solution.

## Project Structure
```
AIT2409110_FinalProject.zip
├── AIT2409110_Cover.pdf  # Project cover with title, members' info, and declaration
├── README.txt             # Instruction manual
├── main.py                # Main script to run the program
├── mnist_cnn_model.h5     # Trained CNN model file
├── /assets/               # Additional resources (e.g., example images, documentation)
└── /src/                  # Source code for model training and GUI implementation
```

## Requirements
- Python 3.8 or later
- TensorFlow 2.x
- NumPy
- Pillow (PIL)
- Tkinter (pre-installed with Python)
- **Optional**: GPU support for TensorFlow (recommended for faster training)

## How to Run
1. Install the required Python libraries:
   ```bash
   pip install numpy tensorflow pillow
   ```
2. Navigate to the directory containing `main.py`.
3. Launch the program by running:
   ```bash
   python main.py
   ```
4. The GUI will open. Use the canvas to draw a digit and click "Predict" to see the result.

## User Instructions
1. **Drawing a Digit**: Use the mouse to draw a digit (0–9) on the canvas.
2. **Predicting the Digit**: Click the "Predict" button to display the digit and its confidence score.
3. **Clearing the Canvas**: Click the "Clear Canvas" button to erase the current drawing and start over.

## Model Training and Performance
- **Dataset**: MNIST, containing 60,000 training and 10,000 test images.
- **Model Architecture**: A deep CNN with multiple layers for feature extraction and classification.
- **Training Enhancements**: Data augmentation using rotation, zoom, shifts, and shear for better performance.
- **Results**: Achieved a test accuracy of approximately [your accuracy]% on the MNIST test set.

## Individual Work and Contributions
I single-handedly managed all aspects of the project, including:
- **Model Development and Training**: Designed and implemented the machine learning model from scratch, ensuring it met the required performance standards.
- **GUI Design and Implementation**: Developed a fully functional and user-friendly graphical interface to interact with the model.
- **Code Optimization and Debugging**: Improved the efficiency of the codebase and rigorously debugged every module to guarantee smooth execution.
- **Documentation and Reporting**: Authored comprehensive documentation and detailed reports to explain the methodology, usage, and outcomes of the project.

Every element of this project reflects my personal effort, commitment, and expertise in delivering a robust solution.

## Acknowledgments
This project was inspired by the MNIST digit recognition challenge. 
