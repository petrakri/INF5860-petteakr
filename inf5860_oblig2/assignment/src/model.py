#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 2 in                                             #
# INF5860 - Machine Learning for Image analysis                                 #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2018.03.01                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1

    dims = conf['layer_dimensions']
    N = len(dims)
    mu = 0;
    params = {}
    for i in range(N-1):
        sig = np.sqrt(2/(dims[i]))
        params['W_{}'.format(i+1)] = np.random.normal(mu, sig, size=[dims[i], dims[i+1]])
        params['b_{}'.format(i+1)] = np.zeros(shape=(dims[i+1],1))

    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 2 a)
    if activation_function == 'relu':
        return np.where(Z >= 0, Z, 0)
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 2 b)
    Z -= np.max(Z)
    return np.exp(Z - np.log(np.sum(np.exp(Z),axis=0)))


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 2 c)
    dims = conf['layer_dimensions']
    L = len(dims)
    Z = np.zeros(shape=(dims[L-1], np.array(X_batch).shape[1]))
    features = {}
    Y_proposed = {}
    # Z = x*w + b
    # A = ReLU(Z)
    # Y_proposed = softmax(A)
    features['A_0'] = X_batch
    for i in range(1, L):
        W = params['W_{}'.format(i)].T
        b = params['b_{}'.format(i)]
        A = features['A_{}'.format(i-1)]
        features['Z_{}'.format(i)] = np.dot(W, A) + b
        features['A_{}'.format(i)] = activation(features['Z_{}'.format(i)], \
         conf['activation_function'])
    Y_proposed = softmax(features['Z_{}'.format(L-1)].copy())
    return Y_proposed, features

def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.
    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels
    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 3
    [n_y, m] = Y_reference.shape

    cost = -(1/m) * np.sum(Y_reference * np.log(Y_proposed))
    prop = np.argmax(Y_proposed, axis=0)
    true = np.argmax(Y_reference, axis=0)
    num_correct = np.sum(prop == true)
    #num_correct = ().count(1)
    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 4 a)
    if activation_function == 'relu':
        return np.where(Z >= 0, 1, 0)
    else:
        print("Error: Unimplemented derivative of activation function: {}", \
        activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation.
        Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l]
                for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the
        parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 4 b)
    grad_params = {}
    dims = conf['layer_dimensions']
    type = conf['activation_function']
    L = len(dims) - 1
    [n_y, m] = Y_proposed.shape
    A = features['A_{}'.format(L-1)]
    onem = np.ones((m,1))

    for i in range(L, 0, -1):
        J = activation_derivative(features['Z_{}'.format(i)], type) * \
         np.dot(params['W_{}'.format(i+1)], J) \
         if i < L else Y_proposed - Y_reference
        A = features['A_{}'.format(i-1)]
        grad_params['grad_W_{}'.format(i)] = 1/m * np.dot(A, J.T)
        grad_params['grad_b_{}'.format(i)] = 1/m * np.dot(J, onem)
        ## CONTINUE HERE. PAGE 64 LECTURE 4


    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        updated_params: Updated parameter dictionary.
    """
    # TODO: Task 5
    updated_params = {}

    L = len(conf['layer_dimensions'])
    l = conf['learning_rate']
    for i in range(1, L):

        grad_b = grad_params['grad_b_{}'.format(i)]
        grad_W = grad_params['grad_W_{}'.format(i)]
        b = params['b_{}'.format(i)]
        W = params['W_{}'.format(i)]
        updated_params['b_{}'.format(i)] = b - l*grad_b
        updated_params['W_{}'.format(i)] = W - l*grad_W




    return updated_params
