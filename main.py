
import sys

import preprocess
import cnn


def main(train):
    # Load all data
    X_train, y_train, X_test, y_test, mapping = preprocess.load_data('data')
    
    cr = cnn.CharRecognizer()

    if train:
        # Train the Convolutional Neural Network
        cr.train_model(X_train, y_train, epochs=10)

        # Save the model to 'emnist-cnn.h5'. It can be loaded afterwards with cr.load_model().
        cr.save_model()
    else:
        # Load a trained model instead of training a new one.
        try:
            cr.load_model()
        except:
            print('[Error] No trained CNN model found.')

    # We can use some keras' Sequential model methods too, like summary():
    cr.model.summary()

    # Use the CNN model to recognize the characters in the test set.
    preds = cr.read_text(X_test, mapping)
    print(preds)

    # Evaluate how well the CNN model is doing. 
    # If it's not good enough, we can try training it with a higher number of epochs.
    cr.evaluate_model(X_train, y_train)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        main(train=True)
    else:
        main(train=False)
