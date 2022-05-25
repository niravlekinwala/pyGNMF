from http.client import EXPECTATION_FAILED
from nis import match
from statistics import covariance
import numpy as np

class internal_functions:
    """This is a class for some miscellaneous functions like 3D-Transpose and to check if the matrix is positive semi-definite.
    """

    def __init__(self, mat):
        self.mat = mat

    def transpose3d(mat):
        """This function returns transpose of a 3D Matrix.

        transpose(X[n, a, b]) ==> X[n, b, a]

        Parameters
        ----------
        mat : ndarray, required
            Input Matrix of size (n, a, b)

        Returns
        -------
        mat_t : ndarray
            Matrix of size (n, b, a)
        """
        if len(mat.shape) == 3:
            runs, dim_1, dim_2 = mat.shape
            mat_t = np.zeros((runs, dim_2, dim_1))
            for i in range(runs):
                mat_t[i][:, :] = mat[i][:, :].T
        elif len(mat.shape) >= 4:
            raise Exception("Size of the input matrix is greater than 3.")
        else:
            mat_t = mat.T

        return mat_t

    def is_pos_def(mat):
        """This function checks if the input SQUARE matrix is positive definite.

        Parameters
        ----------
        mat : ndarray, required
            Input Matrix.

        Returns
        -------
        Depending on several conditions, one of the following statement will print,\\
            1. ``The input matrix is not square`` : If the Matrix is not Square.\\
            2. ``The matrix is not symmetric`` : If the Matrix is not Symmetric.\\
            3. ``The input matrix is Positive Definite`` : The Matrix is Square,
            Symmetric and Positive Definite.\\
            4. ``The input matrix is NOT Positive Definite`` : The Matrix is
            Square, Symmetric but NOT Positive Definite.
        """
        size1, size2 = mat.shape

        if size1 != size2:
            raise Exception("The input matrix is not square")
            return 0

        elif np.sum(mat.T - mat) > 1e-1:
            print("The matrix is not symmetric.")
            return 0

        elif np.sum(np.linalg.eigh(mat)[0] > 0) == size1:
            print ("The input matrix is Positive Definite")
            return 1

        else:
            print("The input matrix is NOT Positive Definite")
            return 0
        
class covariance_matrix_handling:
    """This class deals with the different aspects of handling the covariance matrix
    """
    def __init__(self, mat):
        self.Covariance = Covariance
        self.XMatrix = XMatrix
        self.n_samples, self.m_species = XMatrix.shape
        self.option = option

    def restructure_covariance(Covariance, n_samples, m_species, option):
        """The function is used to restructure the Covariance Matrix
        differently for the update of `G` and `F` Matrices.

        Parameters
        ----------
        Covariance : ndarray
            Size -> nm`x`nm\n
            The original Covariance Matrix. Python flattens the `X` matrix by
            order 'C', i.e., Row elements are stacked one below another.
        n_samples : float
            Number of Sampes.
        m_species : float
            Number of Species.

        Returns
        -------
        CovarianceColumnInverse_Fupd : ndarray
            Size -> nm`x`nm\n
            Covariance matrix structured for the update of `F`.
        CovarianceRowInverse_Gupd : ndarray
            Size -> nm`x`nm\n
            Covariance matrix structured for the update of `G`.
        """
        if option == 'row_stacked':
            # Row Stacked Covariance Matrix -- Update of G
            CovarianceRow = Covariance
            CovarianceRowInverse_Gupd = np.linalg.inv(Covariance)

            # Column Stacking Covariance Matrix -- Update of F
            CovarianceColumn = np.zeros(Covariance.shape)
            indI = np.empty(0, dtype='int')
            for l in range(m_species):
                indI = np.append(indI, np.arange(0+l, (n_samples*m_species)+l, m_species))
            
            indJ = indI
            for a, i in enumerate(indI):
                for b, j in enumerate(indJ):
                    CovarianceColumn[a, b] = CovarianceRow[i, j]

            CovarianceColumnInverse_Fupd = np.linalg.inv(CovarianceColumn)

        elif option == 'column_stacked':
            # Column Stacked Covariance Matrix -- Update of F
            CovarianceColumn = Covariance
            CovarianceColumnInverse_Fupd = np.linalg.inv(Covariance)

            # Column Stacking Covariance Matrix -- Update of G
            CovarianceRow = np.zeros(Covariance.shape)
            indI = np.empty(0, dtype='int')
            for l in range(m_species):
                indI = np.append(indI, np.arange(0+l, (n_samples*m_species)+l, n_samples))

            indJ = indI
            for a, i in enumerate(indI):
                for b, j in enumerate(indJ):
                    CovarianceRow[a, b] = CovarianceColumn[i, j]

            CovarianceRowInverse_Gupd = np.linalg.inv(CovarianceRow)

        return CovarianceColumnInverse_Fupd, CovarianceRowInverse_Gupd

class GNMF_multiplicative:
    """This class is used to implement GNMF Method with
    multiplicative updates and full (dense) Covariance Matrix.

    Description
    -----------
    Generalised Non-Negative Matrix Factorisation (GNMF) is described in this class.  Following are some functions which are part
    of the class,

    `update_F` : function
        This function is used for the update of F.
    `update_G` : function
        This function is used for the update of G.
    `objective_function` : function
        This function is used to compute the value of objective function

    """





