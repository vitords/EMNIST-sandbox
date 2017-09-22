# EMNIST-sandbox


## Setting it up
```
   $ virtualenv EMNIST-sandbox
   $ git clone https://github.com/vitords/EMNIST-sandbox.git
   $ cd EMNIST-sandbox
   $ source bin/activate
   $ pip install -r requirements.txt
```

## EMNIST dataset

Download the binary [b]ByClass[/b] dataset from https://www.nist.gov/itl/iad/image-group/emnist-dataset and extract it to `EMNIST-sandbox/data/`. The directory structure should look like this:
```
   EMNIST-sandbox/
      data/
         emnist-byclass-mapping.txt
         emnist-byclass-test-images-idx3-ubyte
         emnist-byclass-test-labels-idx1-ubyte
         emnist-byclass-train-images-idx3-ubyte
         emnist-byclass-train-labels-idx1-ubyte
      ...
```

## Usage
Usage examples are provided by the `main.py` script.

The script will either train a new model or load an existing trained model and display the model summary. 

It will then recognize the text from the whole test set and evaluate the loss and accuracy of the model on the test data.

To train a new Convolutional Neural Network:
```
   $ python main.py train
```
To load an existing Convolutional Neural Network, simply:
```
   $ python main.py
```

## Jupyter Notebook

A Jupyter Notebook is provided with some investigations on how to deal with the EMNIST dataset. It can be viewed locally by running `$ jupyter notebook` inside the virtual environment, or viewed on GitHub by opening the `EMNIST-sandbox.ipynb` file.
