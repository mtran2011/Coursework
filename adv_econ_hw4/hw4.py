import numpy as np 

def predict_states(y, R, Q, A):
    '''
    Args:
        y (matrix nx1): column matrix, observations of Y
        R (float): scalar variance of y|x
        Q (matrix 2x2): variance of x(k)|x(k-1)
        A (matrix 2x2): mean factor in x(k)|x(k-1)
    Returns:
        matrix: 2xn dimension, the prior mean of x(k) i.e. E[x(k) | y(:k-1)]
    '''
    m_minus = np.matrix([y[0], 0]).T # mean of the prior of x1, in column
    P_minus = np.matrix(np.identity(2))
    n = y.size # number of observations
    H = np.matrix([1,0]) # H is a row

    prior_means = np.matrix(np.empty((2,n)))
    prior_means[:,0] = m_minus

    for k in range(1,n):
        # the update step to calculate the posterior mean of x(k-1)        
        S = np.asscalar(H * P_minus * H.T) + R # S is scalar
        K = P_minus * H.T * (1 / S) # K is a column
        v = y[k-1] - np.asscalar(H * m_minus) # v is scalar

        m = m_minus + K * v # posterior mean of x(k-1)
        P = P_minus - K * S * K.T # posterior variance of x(k-1)

        # now the predict step for x(k) to calculate the prior of x(k)
        m_minus = A * m
        P_minus = A * P * A.T + Q

        prior_means[:,k] = m_minus
    return prior_means

def predict_y(y, R, Q, A):
    '''
    Returns:
        matrix: nx1 dimension for E[y(k) | x(k)]
    '''
    H = np.matrix([1,0])
    predicted_y = H * predict_states(y, R, Q, A)
    return predicted_y.T

def mse(y, R, Q, A):
    return np.linalg.norm(y - predict_y(y, R, Q, A), ord=2)**2 / y.size
    