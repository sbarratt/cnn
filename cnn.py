from abc import ABCMeta, abstractmethod
import gzip
import os, struct
from random import shuffle
from array import array
import pickle

import numpy as np
from tqdm import tqdm

def read(digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images')
        fname_lbl = os.path.join(path, 'train-labels')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images')
        fname_lbl = os.path.join(path, 't10k-labels')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    images = []
    labels = []
    for i in xrange(len(ind)):
        images.append( np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]) )
        labels.append( lbl[ind[i]] )

    return images, labels



class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward_propagation(self, input):
        """Computes output M(I,W) as a function of input I and parameters W."""
        return

    @abstractmethod
    def backward_propagation(self, output_gradient):
        """Computes the input gradient as a function of the output gradient."""
        return

    @abstractmethod
    def gradient_function(self, output_gradient):
        """Computes the weight gradient with respect to the output gradient."""
        return

    @abstractmethod
    def weight_update(self, weight_gradients, learning_rate):
        """"Adds the weight gradients to the weights using some updating rules
        (batch, stochastic, momentum, weight decay, etc)."""
        return


class IdentityLayer(Layer):
    def __init__(self, num_units):
        self.num_units = num_units

    def forward_propagation(self, I):
        """Computes output M(I,W) as a function of input I and parameters W."""
        assert I.shape[0] == self.num_units

        return I

    def backward_propagation(self, output_gradient, I):
        """Computes the input gradient as a function of the output gradient."""
        assert g.shape[0] == self.num_units

        return g

    def gradient_function(self, output_gradient):
        """Computes the weight gradient with respect to the output gradient."""
        assert g.shape[0] == self.num_units

        return np.zeros((self.num_units, self.num_units))

    def weight_update(self, weight_gradients, learning_rate):
        """"Adds the weight gradients to the weights using some updating rules
        (batch, stochastic, momentum, weight decay, etc)."""
        return


class HiddenLayer(Layer):
    def __init__(self, num_input_units, num_output_units, activation_function):
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.W = np.random.normal(loc = 0.0, scale = 0.05, size = (self.num_output_units, self.num_input_units))
        self.activation_function = activation_function

    def forward_propagation(self, I):
        """Computes output M(I,W) as a function of input I and parameters W."""
        assert I.shape[0] == self.num_input_units
        z = self.W.dot(I)

        return self.activation_function(z)

    def backward_propagation(self, output_gradient, I):
        """Computes the input gradient as a function of the output gradient."""
        assert output_gradient.shape[0] == self.num_output_units
        assert I.shape[0] == self.num_input_units

        z = self.W.dot(I)
        J = np.repeat(self.activation_function(z)[None, :], self.num_input_units, axis=0)
        J = J * (1 - J) * self.W.T

        return J.dot(output_gradient)

    def gradient_function(self, output_gradient, I):
        """Computes the weight gradient with respect to the output gradient."""
        assert output_gradient.shape[0] == self.num_output_units, output_gradient.shape[0]
        assert I.shape[0] == self.num_input_units

        #return I[:, None].dot(output_gradient[None, :])
        return output_gradient[:, None].dot(I[None, :])

    def weight_update(self, weight_gradient, learning_rate):
        """"Adds the weight gradients to the weights using some updating rules
        (batch, stochastic, momentum, weight decay, etc)."""
        assert weight_gradient.shape == (self.num_output_units, self.num_input_units)

        self.W -= learning_rate * weight_gradient


class ConvolutionLayer(Layer):
    pass


class NeuralNetwork():
    def __init__(self, layers, error_function):
        self.layers = layers
        self.error_function = error_function

    def output(self, I):
        output = I
        for layer in self.layers:
            output = layer.forward_propagation(output)

        return output

    def predict(self, I):
        output = self.output(I)
        return np.argmax(output)

    def forward_propagation(self, I):
        output = I
        layer_inputs = []
        for layer in self.layers:
            layer_inputs.append(output)
            output = layer.forward_propagation(output)

        return output, layer_inputs

    def weight_update(self, I, O, learning_rate):
        assert I.shape[0] == self.layers[0].num_input_units
        assert O.shape[0] == self.layers[-1].num_output_units

        output, layer_inputs = self.forward_propagation(I)

        output_gradient = (output - O)

        for layer, layer_input in zip(self.layers[::-1], layer_inputs[::-1]):
            weight_gradient = layer.gradient_function(output_gradient, layer_input)
            layer.weight_update(weight_gradient, learning_rate)
            output_gradient = layer.backward_propagation(output_gradient, layer_input)

    def train(self, train_images, train_labels, validation_images, validation_labels):
        learning_rate = 1.0
        num_epochs = 5
        alpha = .3

        for epoch in tqdm(range(num_epochs)):
            # shuffle data
            combined = zip(train_images, train_labels)
            shuffle(combined)

            train_images[:], train_labels[:] = zip(*combined)

            # stochastic gradient descent
            for i in tqdm(range(len(train_images))):
                train_image, train_label = train_images[i], train_labels[i]

                O = np.zeros(10)
                O[train_label] = 1
                I = train_image

                self.weight_update(I, O, learning_rate)

            # print progress on validation set
            print np.sum( np.array([self.predict(x) for x in validation_images]) == validation_labels ) * 1.0, "/", len(validation_labels)

            # update learning rate
            learning_rate *= alpha


if __name__ == '__main__':

    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))
    sigmoid = np.vectorize(sigmoid)

    def error_function(x, y):
        assert x.shape[0] == y.shape[0]
        return 0.5 * np.sum( (x - y) ** 2 )

    ol = HiddenLayer(100, 10, sigmoid)

    inp = np.random.random(100)
    epsilon = 1e-6

    out = ol.forward_propagation(inp)
    out_epsilon = ol.forward_propagation(inp + epsilon)

    grad_e = (out - out_epsilon) / epsilon

    test = np.zeros(10)
    test[0] = 1
    grad = ol.backward_propagation(test, inp)

    import IPython as ipy
    ipy.embed()

    import sys
    sys.exit(0)

    nn = NeuralNetwork([HiddenLayer(28*28, 100, sigmoid), HiddenLayer(100, 10, sigmoid)], error_function)
    inp = np.random.random(28*28)
    # print ("before:", nn.output(inp))
    # for _ in tqdm(range(20)):
    #     nn.weight_update(inp, np.ones(10), 1)
    # print ("after:", nn.output(inp))

    images, labels = read(dataset='training', path='data/raw')

    combined = zip(images, labels)
    shuffle(combined)
    images[:], labels[:] = zip(*combined)

    validation_images, validation_labels = images[:10000], labels[:10000]
    train_images, train_labels = images[10000:], labels[10000:]

    nn.train(train_images, train_labels, validation_images, validation_labels)

    # save weights
    layer_weights = []
    for layer in nn.layers:
        layer_weights.append( layer.W )
    pickle.dump(layer_weights, open('nn.pickle', 'wb'))

    test_images, test_labels = read(dataset='training', path='data/raw')
