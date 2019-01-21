import numpy as np
from numpy import genfromtxt
from numpy.linalg import inv

# Load data in Dataframe with Pandas
my_data = genfromtxt('test_data.csv', delimiter=',')
X=my_data[:, 0:3]
y=my_data[:, 3]
y=np.transpose([y])

###############################
# Closed form linear regression
###############################

w = inv(X.T @ X) @ X.T @ y
print("Closed Form: \r", w)

# Check convergence using P-norm
def compute_Pnorm(w_0, w_1, p):
    tobesummed = np.power((w_1 - w_0), p)
    return np.sum(tobesummed) ** (1 / p)

###########################################
# Linear Regression using gradient descent
###########################################

#Set hyper parameters
eta_0 = 0.5
beta = 0.1
eps = 0.000005

#Initialize w_arr, i and converged
converged = False
i = 1
alpha = eta_0 / (1 + beta * i)
w_arr = np.ones([1,3])  #array containing estimated w at each iteration i

#iterating to converge on w_arr
while converged != True:
    w_i = w_arr[[i-1]].T - 2 * alpha * (X.T @ X @ w_arr[[i-1]].T - X.T @ y)
    w_i = w_i.T
    w_arr = np.vstack((w_arr, w_i))
    if compute_Pnorm(w_arr[i-1], w_arr[i], 2) <= eps:
        converged = True
    #print(w_arr[i])
    i += 1
    alpha = eta_0 / (1 + beta * i)

print("Gradient Descient: \r", w_arr[[-1]].T)

