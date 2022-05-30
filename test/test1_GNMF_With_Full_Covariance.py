try:
    import cupy as np
    print("Using CuPy")
except ImportError as e:
    import numpy as np
    print("Using NumPy")

from scipy import io
from pyGNMF import gnmf_multiplicative_update as gmult
from pyGNMF import nmf_multiplicative_update as mult
from pyGNMF import gnmf_projected_gradient as gproj

data = io.loadmat("pyGNMF_testDataset.mat")
X_matrix = data['conc_with_error']
covariance = data['covariance']
G_init = data['g_init']
F_init = data['f_init']

GMat, FMat, OFunc = gproj.running_method(
    X_matrix,
    covariance,
    G_init = G_init[1],
    F_init = F_init[1],
    option='row_stacked',
    num_fact=7,
    num_init=1,
    max_iter=10000,
    tolerance=1e-6,
    conv_typ='relative',
    conv_num=10,
)

GMat, FMat, OFunc = gmult.running_method(
    X_matrix,
    covariance,
    G_init = G_init[1],
    F_init = F_init[1],
    option='row_stacked',
    num_fact=7,
    num_init=1,
    max_iter=10000,
    tolerance=1e-06,
    conv_typ='relative',
    conv_num=10,
)
"""

GMat, FMat, OFunc = mult.running_method(
    X_matrix,
    G_init = G_init[1],
    F_init = F_init[1],
    num_fact=7,
    num_init=1,
    max_iter=100000,
    tolerance=1e-6,
    conv_typ='relative',
    conv_num=3,
)
"""
