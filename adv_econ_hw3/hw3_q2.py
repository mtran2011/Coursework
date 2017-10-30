import numpy as np 
from sklearn.linear_model import Ridge

class CoordDescent(object):
    '''
    Attributes:
        x (ndarray): an array of shape (n,d) for n points, d features incl constant
        y (ndarray): an array of shape (n,) for n points
        lamda (float): penalizing constant for 1-norm of beta
        beta (ndarray): an array of shape (d,) for d features
    '''
    def run_ridge(x, y, lamda):


    def __init__(self, x, y, lamda):
        self.x = x
        self.y = y
        self.lamda = lamda
        self.beta = 