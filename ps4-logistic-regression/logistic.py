import math

# Some helper functions
def vec_dot(x,y):
    """
    Implements dot product of two vectors
    """
    return sum([x[i] * y[i] for i in range(len(x))])

def sigmoid(x):
    """
    Implements the sigmoid function on scalar
    """
    return 1/(1+math.exp(x))

# Train logistic regression model
def logistic_train(X, y, alpha = 0.1, epsilon = 1e-3, max_iter = 10000):
    """
    Trains logistic regression model using gradient descent on MLE loss
    
    Inputs:
    X - list of m training examples, where each training example is 
        stored as a list, each example has n features
    y - list of training labels
    alpha - learning rate, default value 0.1
    epsilon - tolerance of error for convergence, default value 1e-3
    max_iter - maximum number of iterations, default = 10000
    
    Outputs:
    w - list of parameters of logistic model
    """
    # Store data dimensions
    m = len(X) # number of examples
    n = len(X[0]) # number of features
    
    # Initialize parameters and number of iterations
    w = n*[0]
    w_prev = n*[1]
    it = 0
    
    while sum([abs(w_prev[i]-w[i]) for i in range(n)]) > epsilon:
        it += 1
        if it >= max_iter:
            break   
        for i in range(m):
            w_prev = w
            w = [w[j] + alpha*(sigmoid(vec.dot(w,X[i]))-y[i]) for j in range(n)]
            
    return w
