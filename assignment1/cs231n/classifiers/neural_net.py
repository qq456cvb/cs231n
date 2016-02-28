import numpy as np
import matplotlib.pyplot as plt

class Net(object):
  """docstring for Net"""
  def __init__(self, layer_sizes, std=1e-4):
    self.params = {}
    self.layer_num = len(layer_sizes)
    for x in xrange(len(layer_sizes)-1):
      self.params['W%d' % (x+1)] = std * np.random.randn(layer_sizes[x], layer_sizes[x+1])
      self.params['b%d' % (x+1)] = np.zeros(layer_sizes[x+1])
    
  def loss(self, X, y=None, reg=0.0):
    self.layers = []
    layers = self.layers
    layers.append(X)
    pre_layer = X
    for x in xrange(self.layer_num-2):
      layers.append(np.maximum(0, pre_layer.dot(self.params['W%d' % (x+1)])
        +self.params['b%d' % (x+1)].reshape(1,-1)))
      pre_layer = layers[-1]
    layers.append(pre_layer.dot(self.params['W%d' % (self.layer_num-1)])
      +self.params['b%d' % (self.layer_num-1)])

    # Unpack variables from the params dictionary
    # W1, b1 = self.params['W1'], self.params['b1']
    # W2, b2 = self.params['W2'], self.params['b2']
    # N, D = X.shape
    # H, C = W2.shape
    N, C = layers[0].shape[0], layers[-1].shape[1]

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # mid = np.maximum(0, X.dot(W1)+b1.reshape(1,-1)) # activation
    # scores = mid.dot(W2)+b2.reshape(1,-1)
    scores = layers[-1]
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    exp_score = np.exp(scores)
    exp_score_sum = exp_score.sum(axis=1)
    correct_score = exp_score[np.arange(N), y]
    probability = (correct_score/exp_score_sum).reshape(-1,1)
    loss = -np.log(probability).sum()

    loss /= N
    for x in xrange(self.layer_num-1):
      loss += 0.5 * reg * np.sum(self.params['W%d' % (x+1)]*self.params['W%d' % (x+1)])

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    des = np.tile((-correct_score/np.square(exp_score_sum)).reshape(-1,1), (1,C))
    des[np.arange(N), y] += 1.0/exp_score_sum
    dsoftmax = des * (-np.ones((layers[-2].shape[0], 1))/probability) * np.exp(scores)
    
    dlayer = {}
    dlayer['l%d' % len(layers)] = dsoftmax # store the derivitive of each layer

    for x in reversed(xrange(self.layer_num-1)):
      x += 1
      if x == self.layer_num-1:
        grads['W%d' % x] = layers[-2].T.dot(dlayer['l%d' % len(layers)])
        grads['W%d' % x] /= N
        grads['W%d' % x] += reg * self.params['W%d' % x]

        grads['b%d' % x] = np.ones_like(self.params['b%d' % x].reshape(1,-1)) * dlayer['l%d' % len(layers)]
        grads['b%d' % x] = np.mean(grads['b%d' % x], axis=0).reshape(-1)
      else:
        # first compute dlayer in the reversed order
        dlayer['l%d' % (x+1)] = dlayer['l%d' % (x+2)].dot(self.params['W%d' % (x+1)].T)

        binary = np.zeros_like(layers[x])
        binary[layers[x]>0] = 1
        grads['W%d' % x] = layers[x-1].T.dot(binary * dlayer['l%d' % (x+1)]) # chain rule, compute dmid/dW1 * dscore/dmid * dsoftmax
        grads['W%d' % x] /= N
        grads['W%d' % x] += reg * self.params['W%d' % x]

        # b1
        grads['b%d' % x] = np.ones_like(self.params['b%d' % x].reshape(1,-1)) * binary * dlayer['l%d' % (x+1)]
        grads['b%d' % x] = np.mean(grads['b%d' % x], axis=0).reshape(-1)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads
    
  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      choices = np.random.choice(num_train, batch_size)
      X_batch = X[choices]
      y_batch = y[choices]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      for x in xrange(self.layer_num-1):
        x += 1
        self.params['W%d' % x] -= learning_rate * grads['W%d' % x]
        self.params['b%d' % x] -= learning_rate * grads['b%d' % x]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.argmax(self.loss(X), axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """

    self.layers = []
    layers = self.layers
    layers.append(X)

    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    H, C = W2.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    mid = np.maximum(0, X.dot(W1)+b1.reshape(1,-1)) # activation
    scores = mid.dot(W2)+b2.reshape(1,-1)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    exp_score = np.exp(scores)
    exp_score_sum = exp_score.sum(axis=1)
    correct_score = exp_score[np.arange(N), y]
    probability = (correct_score/exp_score_sum).reshape(-1,1)
    loss = -np.log(probability).sum()

    loss /= N
    loss += 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    des = np.tile((-correct_score/np.square(exp_score_sum)).reshape(-1,1), (1,C))
    des[np.arange(N), y] += 1.0/exp_score_sum
    dsoftmax = des * (-np.ones((mid.shape[0], 1))/probability) * np.exp(scores)
    
    # W2
    grads['W2'] = mid.T.dot(dsoftmax)
    grads['W2'] /= N
    grads['W2'] += reg * W2

    # b2
    grads['b2'] = np.ones_like(b2.reshape(1,-1)) * dsoftmax
    grads['b2'] = np.mean(grads['b2'], axis=0).reshape(-1)

    # W1
    binary = np.zeros_like(mid)
    binary[mid>0] = 1
    grads['W1'] = X.T.dot(binary * dsoftmax.dot(W2.T)) # chain rule, compute dmid/dW1 * dscore/dmid * dsoftmax
    grads['W1'] /= N
    grads['W1'] += reg * W1

    # b1
    grads['b1'] = np.ones_like(b1.reshape(1,-1)) * binary * dsoftmax.dot(W2.T)
    grads['b1'] = np.mean(grads['b1'], axis=0).reshape(-1)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      choices = np.random.choice(num_train, batch_size)
      X_batch = X[choices]
      y_batch = y[choices]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b2'] -= learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % (iterations_per_epoch / 2) == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.argmax(self.loss(X), axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


