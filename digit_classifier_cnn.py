import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np

# --- Configuration ---
IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 10 # 10 epochs is usually sufficient for high accuracy on MNIST

def load_and_preprocess_data():
    """Loads the MNIST dataset and preprocesses the images and labels."""
    print("--- Loading and Preprocessing Data ---")
    
    # Load the MNIST dataset from Keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape the data to fit the CNN input format (28x28x1 for grayscale)
    # The new shape is (number_of_samples, height, width, channels)
    x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
    
    # Convert pixel values from 0-255 to 0-1 (Normalization)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Convert class vectors to binary class matrices (One-Hot Encoding)
    # e.g., digit 5 becomes [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    
    print(f"x_train shape: {x_train.shape}")
    print(f"{x_train.shape[0]} train samples")
    print(f"{x_test.shape[0]} test samples")
    
    return x_train, y_train, x_test, y_test

def build_cnn_model():
    """Defines the Convolutional Neural Network architecture."""
    print("--- Building CNN Model ---")
    
    model = Sequential([
        # 1. Convolutional Layer (Feature Extraction)
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE, name='Conv1'),
        
        # 2. Convolutional Layer (Deeper Feature Extraction)
        Conv2D(64, (3, 3), activation='relu', name='Conv2'),
        
        # 3. Pooling Layer (Downsampling/Reducing Dimensionality)
        MaxPooling2D(pool_size=(2, 2), name='MaxPool1'),
        
        # 4. Dropout (Regularization to prevent Overfitting)
        Dropout(0.25, name='Dropout1'),
        
        # 5. Flatten Layer (Convert 2D feature maps to 1D vector for Dense layers)
        Flatten(name='Flatten'),
        
        # 6. Dense Layer (Hidden Layer for Classification)
        Dense(128, activation='relu', name='Dense1'),
        
        # 7. Dropout
        Dropout(0.5, name='Dropout2'),
        
        # 8. Output Layer
        # Softmax activation gives a probability distribution over the 10 classes
        Dense(NUM_CLASSES, activation='softmax', name='Output')
    ])
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy', # Appropriate loss for multi-class classification
        optimizer='adam',                # Efficient optimizer
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    """Trains the model and evaluates its performance on the test set."""
    print("--- Training Model ---")
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    
    print("--- Evaluating Model ---")
    # Evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print("-" * 30)
    print(f"Test Loss: {score[0]:.4f}")
    print(f"Test Accuracy: {score[1]*100:.2f}%")
    print("-" * 30)

    # Optional: Save the trained model
    model.save('mnist_cnn_classifier.h5')
    print("Model saved to 'mnist_cnn_classifier.h5'")
    
    return history

if __name__ == '__main__':
    # 1. Load and prepare data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # 2. Build the model
    model = build_cnn_model()
    
    # 3. Train and evaluate
    train_and_evaluate(model, x_train, y_train, x_test, y_test)

    print("\nScript finished.")
    print("You can now load 'mnist_cnn_classifier.h5' for predictions.")