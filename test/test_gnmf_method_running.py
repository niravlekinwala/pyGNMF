## Using pyGNMF
from pyGNMF import gnmf_projected_gradient as gproj
from pyGNMF import gnmf_multiplicative_update as gmult
from pyGNMF import nmf_multiplicative_update as nmfmult
from scipy import io
import numpy as np

data = io.loadmat("IllustrativeExample.mat")

X_error = data['X_error']
covariance = data['covariance']
G_init = data['G_init']
F_init = data['F_init']
num_fact = 3

GMat, FMat, OFunc = gproj.running_method(
    X_matrix = X_error,
    covariance = covariance,
    G_init = G_init,
    F_init = F_init,
    option='row_stacked',
    num_fact=num_fact,
    num_init=1,
    alpha_init_G=1e-5,
    alpha_init_F=1e-5,
    max_iter=500000,
    tolerance=1e-6,
    conv_typ='relative',
    conv_num=3,
)

GMat, FMat, OFunc = gmult.running_method(
    X_matrix = X_error,
    covariance = covariance,
    G_init = G_init,
    F_init = F_init,
    option='row_stacked',
    num_fact=num_fact,
    num_init=1,
    max_iter=500000,
    tolerance=1e-6,
    conv_typ='relative',
    conv_num=3,
)

GMat, FMat, OFunc = nmfmult.running_method(
    X_matrix = X_error,
    G_init = G_init,
    F_init = F_init,
    option='row_stacked',
    num_fact=num_fact,
    num_init=1,
    max_iter=500000,
    tolerance=1e-6,
    conv_typ='relative',
    conv_num=3,
)

