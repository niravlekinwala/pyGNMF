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
X_matrix = np.array(data['conc_with_error'])
covariance = np.array(data['covariance'])
G_init = np.array(data['g_init'])
F_init = np.array(data['f_init'])
GMat, FMat, OFunc = gproj.running_method(
    X_matrix,
    covariance,
    G_init = G_init[0],
    F_init = F_init[0],
    option='row_stacked',
    num_fact=7,
    num_init=1,
    alpha_init_G=1,
    alpha_init_F=1,
    max_iter=100,
    tolerance=1e-6,
    conv_typ='relative',
    conv_num=3,
)

"""
GMat, FMat, OFunc = gmult.running_method(
    X_matrix,
    covariance,
    G_init = 'random',
    F_init = 'random',
    option='row_stacked',
    num_fact=3,
    num_init=1,
    max_iter=10000,
    tolerance=1e-6,
    conv_typ='relative',
    conv_num=3,
)

GMat, FMat, OFunc = mult.running_method(
    X_matrix,
    G_init = 'random',
    F_init = 'random',
    num_fact=3,
    num_init=1,
    max_iter=100000,
    tolerance=1e-6,
    conv_typ='relative',
    conv_num=3,
)
"""
