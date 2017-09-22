
import numpy as np

from mnist import MNIST
from keras.utils import np_utils


def load_data(path):
    """Load data from the EMNIST dataset.

    All the data files should be using the original file names as given by
    the EMNIST website (https://www.nist.gov/itl/iad/image-group/emnist-dataset).

    Args:
        path (str): Directory containing all data files.

    Returns:
        Train and test data arrays with their respective label arrays
        and label mapping.
    """

    # Read all EMNIST test and train data
    mndata = MNIST(path)

    X_train, y_train = mndata.load(path + '/emnist-byclass-train-images-idx3-ubyte', 
                                   path + '/emnist-byclass-train-labels-idx1-ubyte')
    X_test, y_test = mndata.load(path + '/emnist-byclass-test-images-idx3-ubyte', 
                                 path + '/emnist-byclass-test-labels-idx1-ubyte')

    # Read mapping of the labels and convert ASCII values to chars
    mapping = []

    with open(path + '/emnist-byclass-mapping.txt') as f:
        for line in f:
            mapping.append(chr(int(line.split()[1])))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test, mapping


def normalize(array):
    """Normalize an array with data in an interval of [0, 255] to [0, 1].
    

    Args:
        array (numpy.ndarray): Data array to be normalized.

    Returns:
        Array with all values inside the interval [0, 1].
    """
    array = array.astype('float32')
    array /= 255

    return array


def reshape_for_cnn(array, color_channels, img_width, img_height):
    """Reshape the image data to be used in a Convolutional Neural Network.

    Args:
        array (numpy.ndarray): Data to be reshaped.

    Returns:
        Reshaped array containing all original data.

    """
    return  array.reshape(array.shape[0], color_channels, img_width, img_height)

def preprocess_labels(array, nb_classes):
    """Perform an "one-hot encoding" of a label array (multiclass).

    Args:
        array (numpy.ndarray): Array of labels (multiclass).
        nb_classes: Total number of classes.

    Returns:
        One-hot encoded label array.
    """
    return np_utils.to_categorical(array, nb_classes)
