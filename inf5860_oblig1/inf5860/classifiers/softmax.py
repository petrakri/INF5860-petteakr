import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    [D, C] = W.shape
    [N, D]= X.shape
    
    # denum = p matrisen i gradienten
    # lambda -> reg   theta -> W
    # m -> N    y(i) -> y[i]
    # k -> C    theta_j -> W[:,j]
    # l -> C    x(i) = X[i,:]
    
    for i in range(N):
        
        denum = 0
        
        for l in range(C):
            
            denum += np.exp(np.dot(W[:,l],X[i,:]))
        
        for j in range(C):
            
            q = np.exp(np.dot(W[:,j],X[i,:]))/denum
            loss += (y[i] == j) * np.log(q)
            dW[:,j] += X[i,:]*((y[i] == j) - q)
    
    
    # summe regulariseringen til slutt etter alle løkker R += W[i,j]**2 == np.sum(W**2)
    loss = -(loss/N) + (reg/2)*np.sum(W**2)
    dW = -(dW/N) + reg*W
    
    pass
      
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    [D, C] = W.shape
    [N, D]= X.shape
    
    # denum = p matrisen i gradienten
    # lambda -> reg   theta -> W
    # m -> N    y(i) -> y[i]
    # k -> C    theta_j -> W[:,j]
    # l -> C    x(i) = X[i,:]

    # Calculate scores, dot product of X and W
    scores = np.dot(X,W)
    
    # Create a p for the true distribution, where is y the correct class
    p = np.zeros_like(scores)
    p[np.arange(N), y] = 1
    
    # shifting, highest value is 0, avoiding numeric stability
    scores -= np.max(scores)
    
    # Softmax function
    q = np.exp(scores) / np.sum(np.exp(scores),axis=1).reshape(N,1) # q for log(q)
    
    # Loss function
    loss = -np.sum(p*np.log(q))
    
    dW = np.dot(X.T,(p - q))
    
    # Skalerer og legger til generalisering
    loss = loss/N + (reg/2)*np.sum(W**2)
    dW = -(dW/N) + reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

