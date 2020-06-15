import numpy as np

from layers import (FullyConnectedLayer, ReLULayer,
                    softmax_with_cross_entropy, l2_regularization,
                    softmax)


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.FCL1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLu = ReLULayer()
        self.FCL2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        prms = self.params()
        for p in prms:
            prms[p].grad = np.zeros_like(prms[p].value)

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        preds = self.FCL2.forward(
                    self.ReLu.forward(
                        self.FCL1.forward(X)))

        (loss, dprediction) = softmax_with_cross_entropy(preds, y)

        d_fcl2 = self.FCL2.backward(dprediction)
        d_relu = self.ReLu.backward(d_fcl2)
        d_fcl1 = self.FCL1.backward(d_relu)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for p in prms:
            (loss_l2, grad_l2) = l2_regularization(prms[p].value, self.reg)
            loss += loss_l2
            prms[p].grad += grad_l2

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        probs = softmax(self.FCL2.forward(
                           self.ReLu.forward(
                                self.FCL1.forward(X))))

        pred = np.argmax(probs, axis=1).astype(np.int)

        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        result['W1'] = self.FCL1.W
        result['B1'] = self.FCL1.B

        result['W2'] = self.FCL2.W
        result['B2'] = self.FCL2.B

        return result
