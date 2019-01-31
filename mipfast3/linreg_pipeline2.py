# Importing all relevant packages
import json
from implement_linreg import gd_lin_reg, cf_lin_reg
from implement_linreg import compute_mse
from proj1_data_loading import transform
import matplotlib.pyplot as plt
import time
start_time = time.time()


# Load json data file into data numpy array
with open("proj1_data.json") as fp:
    data = json.load(fp)

# Splitting data into training, validation and testing datasets
training = data[0:10000] #the list contains elements from position 0 to position 9999, total 10000
validation = data[10000:11000] #the list contains elements from position 10000 to position 10999, total 1000
testing = data[11000:12000] # the list contains elements from position 11000 to position 11999, total 1000

# Splitting each data set into a feature matrix X and a target vector y
X_train,y_train = transform(training)
X_val,y_val = transform(validation)

#X_test,y_test = transform(testing)
X_train = X_train.astype('int64')
X_val = X_val.astype('int64')
#X_test = X_test.astype('int64')
y_train = y_train.reshape(10000,1)
y_val = y_val.reshape(1000,1)
#y_test = y_test.reshape(1000,1)


# Implementing linear regression using gradient descent:
eta_0 = 0.00000001
beta = 0
eps = 0.0001
#print("eta_0:", eta_0)
print("eta_0:",eta_0 ,";","beta:",beta ,";","eps:",eps)
w_gd = gd_lin_reg(X_train, y_train, eta_0, beta, eps)
MSE_gd_train = compute_mse(X_val, w_gd, y_val)
#print("Gradient Descent: \r", w_gd)

print("MSE: ", MSE_gd_train)

#y_predict2 = X_train @ w_gd
#plt.plot(y_train, y_predict2, 'bo', markersize=0.5)
#plt.xlabel ('y_train')
#plt.ylabel ('y_predict2')
#plt.xlim(-7.5, 10)
#plt.ylim(-2.5, 10)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.show()
#plt.clf()

#print("--- %s seconds ---" % (time.time() - start_time))

