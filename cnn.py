
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense


class CharRecognizer(object):
    """Handwritten character recognition with a Convolutional Neural Network."""

    def __init__(self):
        self.model = None

    def train_model(self, X, y, batch_size=128, epochs=10):
        """Train a Convolutional Neural Network to recognize handwritten characters.
        
        Args:
            X (numpy.ndarray): Training data (EMNIST ByClass dataset)
            y (numpy.ndarray): Labels of the training data.
            batch_size (int): How many images the CNN should use at a time.
            epochs (int): How many times the data should be used to train the model.
        """
        self.build_model()
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs)
    
    def evaluate_model(self, X, y):
        """Evaluate the loss and accuracy of the trained model.

        Args:
            X (numpy.ndarray): Test data.
            y (numpy.ndarray): Labels of the test data.
        """
        score = self.model.evaluate(X, y)
        print('Loss:', score[0])
        print('Accuracy:', score[1])

    def save_model(self):
        """Save the trained model to a file."""
        self.model.save('emnist-cnn.h5', overwrite=True)
        
    def load_model(self):
        """Load a trained model from a file."""
        self.model = load_model('emnist-cnn.h5')

    def read_text(self, data, mapping):
        """Identify handwritten characters in images.

        Args:
            data (numpy.ndarray): An array containing the data of the images to be recognized.
            mapping (dict): Label mapping to convert from class to character.

        Returns:
            text (str): Text predicted from the handwritten characters. 
        """
        preds = self.model.predict(data)
        preds = np.argmax(preds, axis=1)
        return ''.join(mapping[x] for x in preds)

    def build_model(self, nb_classes=62, nb_filters=32, kernel_size=(5, 5), pool_size=(2, 2), input_shape=(1, 28, 28)):
        """Build a Convolutional Neural Network model to recognize handwritten characters in images.

        Args:
            nb_classes (int): The number of classes in the EMNIST dataset.
            nb_filters (int): Number of convolutional filters to be used.
            kernel_size (tuple(int, int)):  Size of the kernel (group of weights shared over the image values).
            pool_size (tuple(int, int)): Downscale factor for the MaxPooling2D layer.
            input_shape (tuple(int, int, int)): Shape of the images as (# of color channels, width, height).
        """
        self.model = Sequential()
        self.model.add(Convolution2D(int(nb_filters / 2), kernel_size, padding='valid',
                                input_shape=input_shape, activation='relu',
                                kernel_initializer='he_normal', data_format = 'channels_first'))
        self.model.add(MaxPooling2D(pool_size=pool_size))
        self.model.add(Convolution2D(nb_filters, kernel_size, activation='relu', 
                                kernel_initializer='he_normal', data_format = 'channels_first'))
        self.model.add(MaxPooling2D(pool_size=pool_size))
        
        self.model.add(Flatten())
        self.model.add(Dense(250, activation='relu', kernel_initializer='he_normal'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(125, activation='relu', kernel_initializer='he_normal'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes, activation='softmax', kernel_initializer='he_normal'))
    
        self.model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy']) 
