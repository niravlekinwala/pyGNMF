import numpy as np
from scipy import io
from pyGNMF import gnmf_multupd_with_cov as gmult
from pyGNMF import gnmf_projgrad_with_cov as gproj

data = io.loadmat("pyGNMF_testDataset.mat")
X_matrix = data['conc_with_error']
covariance = data['covariance']
G_init = data['g_init']
F_init = data['f_init']

GMat, FMat, OFunc = gmult.running_method(
    X_matrix,
    covariance,
    G_init = G_init,
    F_init = F_init,
    option='row_stacked',
    num_factors=7,
    num_init=1,
    max_iter=50,
    tolerance=0.2,
    convergence_type='relative',
    convergence_number=10,
)
"""

GMat, FMat, OFunc = gproj.running_method(
    X_matrix,
    covariance,
    G_init = G_init,
    F_init = F_init,
    beta=0.1,
    sigma=0.0001,
    alpha_init_G=1,
    alpha_init_F=1,
    option='row_stacked',
    num_factors=7,
    num_init=1,
    max_iter=500,
    tolerance=1e-02,
    convergence_type='relative',
    convergence_number=10,
)
"""
