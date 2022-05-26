import numpy as np
from scipy import io

#from gnmf_projgrad_with_cov import running_method
#from nmf_multupd import running_method
import pyGNMF

data = io.loadmat("SimulatedProblemData_10x100.mat")
init = io.loadmat("NewInitialGuess.mat")

XMatrix = data['conc_with_error'][0]
n_samples, m_species = XMatrix.shape
Covariance = data['covariance']

GInit = init['g_init'][0][0:5]
FInit = init['f_init'][0][0:5]
#GMat, FMat, OFunc = gnmf_multupd_with_cov.running_method(XMatrix, Covariance, option = 'row_stacked',num_factors=3, convergence_type='relative', tolerance = 1e-6, max_iter = 500000)

GMat, FMat, OFunc = pyGNMF.gnmf_multupd_with_cov.running_method(XMatrix,
                                        G_init = GInit,
                                        F_init = FInit,
                                        covariance = Covariance,
                                        option = 'row_stacked',
                                        num_init = 5,
                                        num_factors=7,
                                        convergence_type='relative',
                                        tolerance = 1e-2,
                                        max_iter = 1000)

"""
GMatn, FMatn, OFuncn = nmf.running_method(XMatrix,
                                        G_init = GInit,
                                        F_init = FInit,
                                        num_init = 5,
                                        num_factors=7,
                                        convergence_type='relative',
                                        tolerance = 1e-6,
                                        max_iter = 15)
"""


#GMat, FMat, OFunc = nmf_multupd.running_method(XMatrix, num_factors=3, convergence_type='relative', tolerance = 1e-6, max_iter = 500000)
