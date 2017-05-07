import numpy as np

def nn_forward(x,theta1,theta2):
    """
    Perform forward calculations on a two layer feedforward neural network
    
    Inputs:
    x - input array of dimension 4
    theta1 - weight matrix for first layer of dimensions 2 by 4
    theta2 - weight matrix for second layer of dimensions 3 by 2
    """
    h = 1 / (1 + np.exp(-np.dot(theta1, x)))
    y = 1 / (1 + np.exp(-np.dot(theta2, h)))
    
    return 1/(1 + np.exp(-y))
