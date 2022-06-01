import numpy as np

## Generating the dataset
num_rows, num_cols, num_fact = 20, 6, 3
G_true = np.random.rand(num_rows, num_fact)
F_true = np.random.rand(num_fact, num_cols)
X_true = G_true@F_true

## Generating Covaraince
a = np.random.randn(num_rows*num_cols, num_rows*num_cols)
covariance = 1e-4*(a.T@a)
cov_inv = np.linalg.inv(covariance)

error_vector = np.random.multivariate_normal(np.zeros(num_rows*num_cols), covariance)

X_error = X_true + error_vector.reshape(num_rows, num_cols)

#from scipy import io
#io.savemat("IllustrativeExample.mat", {
#    "G_true":G_true,
#    "F_true":F_true,
#    "X_true":X_true,
#    "X_error":X_error,
#    "covariance":covariance,
#})
## Using pyGNMF
from pyGNMF import gnmf_projected_gradient as gproj
from pyGNMF import gnmf_multiplicative_update as gmult
from scipy import io
import time

num_rows = 10
num_cols = 10
num_fact = 6

time_per_iteration = []

for i in range(5, 55, 5):
    G_true = np.random.rand(num_rows*i, num_fact)
    F_true = np.random.rand(num_fact, num_cols)
    X_true = G_true@F_true

    ## Generating Covaraince
    a = np.random.randn(num_rows*i*num_cols, num_rows*i*num_cols)
    covariance = 1e-4*(a.T@a)
    error_vector = np.random.multivariate_normal(np.zeros(num_rows*i*num_cols), covariance)

    X_error = X_true + error_vector.reshape(num_rows*i, num_cols)


    #data = io.loadmat("IllustrativeExample.mat")

    X_error = X_error
    covariance = covariance
    num_fact = 6
    start_time = time.time()
    GMat, FMat, OFunc = gproj.running_method(
    X_matrix = X_error,
    covariance = covariance,
    G_init = 'random',
    F_init = 'random',
    option='row_stacked',
    num_fact=6,
    num_init=1,
    alpha_init_G=1e-5,
    alpha_init_F=1e-5,
    max_iter=100,
    tolerance=1e-16,
    conv_typ='relative',
    conv_num=3)
    end_time = time.time()
    time_per_iteration.append((end_time-start_time)/100)
    print("Time per iteration:", (end_time-start_time)/100)




