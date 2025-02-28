import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps

# Load and preprocess the MNIST dataset
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Reshape and normalize
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255.0
    
    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

# Build an improved CNN model
def build_model():
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Flatten and dense layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    # Use a lower learning rate and add learning rate decay
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Create data augmentation generator
def create_data_generator():
    return ImageDataGenerator(
        rotation_range=10,      # Random rotation between 0-10 degrees
        width_shift_range=0.1,  # Horizontal shift
        height_shift_range=0.1, # Vertical shift
        zoom_range=0.1,         # Zoom in/out
        shear_range=0.1,        # Shear transformation
        horizontal_flip=False   # No horizontal flipping for digit recognition
    )

# Train the model with data augmentation
def train_model(model, x_train, y_train, x_test, y_test):
    # Create data generator
    datagen = create_data_generator()
    
    # Fit the model with data augmentation
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=20,  # Increased epochs
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) // 64,
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    
    # Save the model
    model.save("mnist_cnn_model.h5")
    
    return history

# Create the Tkinter application for digit recognition
class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Digit Recognizer")
        
        # Canvas for drawing
        self.canvas = tk.Canvas(master, width=280, height=280, bg='white')
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_line)
        
        # Predict button
        self.button_predict = tk.Button(master, text="Predict", command=self.predict)
        self.button_predict.pack(pady=5)
        
        # Clear button
        self.button_clear = tk.Button(master, text="Clear Canvas", command=self.clear_canvas)
        self.button_clear.pack(pady=5)
        
        # Create a white image
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
    
    def paint(self, event):
        x, y = event.x, event.y
        # Draw on canvas
        self.canvas.create_oval(x-10, y-10, x+10, y+10, 
                                 fill='black', outline='black')
        # Draw on image
        self.draw.ellipse([x-10, y-10, x+10, y+10], fill=0)
    
    def reset_line(self, event=None):
        # This helps create smoother lines when drawing
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        # Clear the canvas
        self.canvas.delete("all")
        # Reset the image to white
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
    
    def predict(self):
        # Prepare the image for prediction
        img_array = np.array(self.image)
        
        # Resize the image to 28x28 using LANCZOS filter
        img_resized = Image.fromarray(img_array).resize((28, 28), Image.LANCZOS)
        
        # Invert the colors (white background to black)
        img_array = ImageOps.invert(img_resized)
        
        # Convert to grayscale and normalize
        img_array = np.array(img_array).reshape((28, 28, 1)).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Load the model and predict
        model = tf.keras.models.load_model("mnist_cnn_model.h5")
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        
        # Show the prediction result
        messagebox.showinfo("Prediction", 
                             f"Predicted digit: {predicted_class}\n"
                             f"Confidence: {confidence*100:.2f}%")

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build and train the model
    model = build_model()
    train_model(model, x_train, y_train, x_test, y_test)
    
    # Start the Tkinter application
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


    # Evaluate the model and generate metrics
    def generate_analysis(model, x_test, y_test):
        # Predict on test data
        predictions = model.predict(x_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))

        # Plot Confusion Matrix
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        disp.plot(ax=plt.gca(), cmap='Blues', colorbar=False)
        plt.title("Confusion Matrix")

        # Prediction Confidence Bar Chart
        confidence_percentages = np.max(predictions, axis=1) * 100
        bins = np.arange(0, 110, 10)

        plt.subplot(1, 2, 2)
        plt.hist(confidence_percentages, bins=bins, edgecolor='black', color='skyblue')
        plt.xticks(bins)
        plt.xlabel("Confidence Percentage (%)")
        plt.ylabel("Frequency")
        plt.title("Prediction Confidence Distribution")

        plt.tight_layout()
        plt.show()

    generate_analysis(model, x_test, y_test)


    # Plot training and validation accuracy over epochs
    def plot_accuracy(history):
        # Extract accuracy values
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        epochs = range(1, len(train_acc) + 1)

        # Plot accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r--', label='Validation Accuracy')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Build and train the model
    model = build_model()
    history = train_model(model, x_train, y_train, x_test, y_test)

    # Generate analysis
    generate_analysis(model, x_test, y_test)

    # Plot accuracy graph
    plot_accuracy(history)
