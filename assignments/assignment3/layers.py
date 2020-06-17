import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return (loss, grad)


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    probs = predictions.copy()

    if predictions.ndim == 1:
        probs -= np.max(probs)
        probs = np.exp(probs)
        probs = probs / np.sum(probs)
    else:
        probs -= np.max(probs,
                        axis=1).reshape((probs.shape[0], 1))

        probs = np.exp(probs)
        probs = probs / np.sum(probs, axis=1).reshape((probs.shape[0], 1))

    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if type(target_index) is int:
        loss = -np.log(probs[target_index])
    else:
        loss = -np.sum(np.log(
            probs[np.arange(target_index.size), target_index]
        )) / target_index.size

    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    dprediction = softmax(predictions)
    loss = cross_entropy_loss(dprediction, target_index)

    if type(target_index) is int:
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(target_index.size), target_index] -= 1
        dprediction /= target_index.size

    return (loss, dprediction)


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        return np.where(X > 0, X, 0)

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = d_out * np.where(self.X > 0, 1, 0)

        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        result = np.dot(self.X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_input = np.dot(d_out, np.transpose(self.W.value))
        self.W.grad += np.dot(np.transpose(self.X), d_out)
        self.B.grad += np.dot(np.ones((1, d_out.shape[0])), d_out)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels))
        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        (batch_size, height, width, channels) = X.shape
        self.X = X
        self.X_pad = np.pad(X, ((0, 0),
                                (self.padding, self.padding),
                                (self.padding, self.padding),
                                (0, 0)),
                            'constant', constant_values=(0))

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1

        result = np.zeros((batch_size, out_height, out_width,
                           self.out_channels))

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        nsize = self.filter_size * self.filter_size * self.in_channels

        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                inp = self.X_pad[
                        :,
                        y:y + self.filter_size,
                        x:x + self.filter_size,
                        :].reshape((batch_size, nsize))

                w = self.W.value.reshape((nsize, self.out_channels))

                result[:, y, x, :] = np.dot(inp, w) + self.B.value

        return result

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        (batch_size, height, width, channels) = self.X.shape
        (_, out_height, out_width, out_channels) = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        d_input = np.zeros_like(self.X_pad)
        nsize = self.filter_size * self.filter_size * self.in_channels

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                X_slice = self.X_pad[:,
                                     y:y + self.filter_size,
                                     x:x + self.filter_size,
                                     :,
                                     np.newaxis]

                grad = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :]

                self.W.grad += np.sum(grad * X_slice, axis=0)

                d_input[:, y:y + self.filter_size,
                        x:x + self.filter_size, :] += np.sum(
                            self.W.value * grad, axis=-1)

        self.B.grad += np.sum(d_out, axis=(0, 1, 2))

        d_input = d_input[:, self.padding:self.padding + height,
                          self.padding:self.padding + width, :]

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        (batch_size, height, width, channels) = X.shape
        self.X = X.copy()

        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        result = np.zeros((batch_size, out_height, out_width,
                           channels))

        for y in np.arange(0, out_height, self.stride):
            for x in np.arange(0, out_height, self.stride):
                result[:, y, x, :] = np.amax(
                    X[:, y:y+self.stride, x:x+self.stride, :],
                    axis=(1, 2))

        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        (batch_size, height, width, channels) = self.X.shape
        (_, out_height, out_width, _) = d_out.shape

        d_input = np.zeros_like(self.X)

        for y in np.arange(0, out_height, self.stride):
            for x in np.arange(0, out_height, self.stride):
                x_pool = self.X[
                    :,
                    y:y+self.stride,
                    x:x+self.stride,
                    :]
                mask = (x_pool == np.amax(x_pool, axis=(1, 2))[:,
                                                               np.newaxis,
                                                               np.newaxis,
                                                               :])

                d_input[:,
                        y:y+self.pool_size,
                        y:y+self.pool_size,
                        :] += mask * d_out[:, y, x, :][:, np.newaxis,
                                                       np.newaxis, :]

        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        (batch_size, height, width, channels) = X.shape
        self.X_shape = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        result = X.reshape((batch_size, height * width * channels))
        return result

    def backward(self, d_out):
        # TODO: Implement backward pass
        d_input = d_out.reshape(self.X_shape)
        return d_input

    def params(self):
        # No params!
        return {}
