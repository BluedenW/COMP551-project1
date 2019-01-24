import numpy as np
from numpy.linalg import inv

#Funtion for obtaining closed-Form Linear Regression solution
def cf_lin_reg(X, y):
    w = inv(X.T @ X) @ X.T @ y
    return w

###########################################
# Linear Regression using gradient descent
###########################################

# Check convergence using P-norm
def compute_Pnorm(w_0, w_1, p):
    tobesummed = np.power((w_1 - w_0), p)
    return np.sum(tobesummed) ** (1 / p)

# Function for calculating the mean squared error
def compute_mse(X, w, y):
    mse = ((X @ w - y) ** 2).mean(axis=None)
    return mse

def gd_lin_reg(X, y, eta_0, beta, eps):
    #Initialize w_arr, i and converged
    converged = False
    i = 1
    alpha = eta_0 / (1 + beta * i)
    w_arr = np.ones([1,len(X[0])])  #array containing estimated w at each iteration i

    #iterating to converge on w_arr
    while converged != True:
        w_i = w_arr[i-1] - 2 * alpha * (X.T @ X @ w_arr[i-1] - X.T @ y)
        w_i = w_i.T
        w_arr = np.vstack((w_arr, w_i))
        if compute_Pnorm(w_arr[i-1], w_arr[i], 2) <= eps:
            converged = True
        i += 1
        alpha = eta_0 / (1 + beta * i)
        print(w_arr[-1].T)
        print(compute_Pnorm(w_arr[i-2], w_arr[i-1], 2))
     #   print(np.shape(w_arr[-1]))
    return w_arr[-1]

