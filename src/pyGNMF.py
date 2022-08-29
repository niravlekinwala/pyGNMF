import numpy as np ## Change 'numpy' with 'cupy' to use GPU
import warnings
warnings.filterwarnings("error")
from tqdm import tqdm
from rich.progress import Progress

##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
class internal_functions:
    """This is a class for some miscellaneous functions like 3D-Transpose and to check if the matrix is Positive Definiteness.
    """

    def transpose_3d(mat):
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
        """This function checks if the input SQUARE matrix is positive Definite.

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
        mat = np.array(mat)
        size_1, size_2 = mat.shape
        is_pos_def_key = 0

        ## CONDITION NUMBER -- WARNING -- NOT STOP CODE
        condition_number = np.linalg.cond(mat)
        if condition_number > 1/np.finfo('float').eps:
            print("`WARNING`:The condition number of the covariance matrix {}. The results may not be accurate.".format(condition_number, 1/np.finfo('float').eps))

        ## DIFFERNENCE
        diff = np.linalg.norm(mat@np.linalg.inv(mat) - np.eye(mat.shape[0]), 'fro')
        print("The Frobenius Norm of the (I - Σ@Σ^(-1)) is: {}".format(diff))


        if size_1 != size_2:
            raise Exception("C1: The input matrix is not square")

        elif np.sum(np.linalg.norm((mat.T - mat), 'fro')) > 1e-6:
            print("C2: The covariance Matrix is NOT Symmetric.")
            return is_pos_def_key

        elif np.sum(np.linalg.eigh(mat)[0] > 0) < size_1:
            print ("C3: The covariance Matrix is NOT Positive Definite")
            is_pos_def_key = 0
            return is_pos_def_key

        else:
            print("C4: The covariance Matrix is Positive Definite")
            is_pos_def_key = 1
            return is_pos_def_key



    def checking_initial_guess(G_init, F_init, n_numrows, m_numcols, p_numfact, num_init):
        """_summary_

        Parameters
        ----------
        G_init : ndarray or str
            Size -> n `x` k
            Initial value for G Matrix.
        F_init : ndarray or str
            Size -> k `x` m
            Initial value for F Matrix.
        n_numrows : int
            Number of rows in the X matrix.
        m_numcols : int
            Number of columns in the X matrix.
        num_init : int
            Number of initialisations for a dataset under consideration.

        Returns
        -------
        G_init : ndarray or str
            Size -> n `x` k
            Initial value for G Matrix.
        F_init : ndarray or str
            Size -> k `x` m
            Initial value for F Matrix.
        """

        if type(G_init) == str and type(F_init) == str and G_init.upper() == "RANDOM" and F_init.upper() == "RANDOM":
             ## Initial guesses need to be randomly generated.
            print("{} option is selected for both G_init and F_init. Generating initial guesses randomly".format(G_init.upper()))
            G_init = np.random.rand(num_init, n_numrows, p_numfact)
            F_init = np.random.rand(num_init, p_numfact, m_numcols)
        else:
            ## CHECKING FOR NEGATIVE VALUES
            if np.sum(G_init < 0) != 0 and np.sum(F_init < 0) != 0:
                raise ValueError("There are negative values in initial guess. Make sure to initialise with positive values.")

            if num_init == 1 and len(G_init.shape) != 3 and len(F_init.shape) != 3:
                G_init = G_init.reshape(num_init, n_numrows, p_numfact)
                F_init = F_init.reshape(num_init, p_numfact, m_numcols)
            elif num_init >= 2 and len(G_init.shape) == 3 and len(F_init.shape) == 3 and G_init.shape[0]!=num_init and F_init.shape[0]!=num_init:
                raise ValueError("Make sure that the size of initial guesses for G and F is sizes {} x {} x {} and {} x {} x {} respectively.".format(num_init, n_numrows, p_numfact, num_init, p_numfact, m_numcols))

        return G_init, F_init

class covariance_matrix_handling:
    """This class deals with the different aspects of handling the covariance matrix
    """

    def restructure_covariance_inverse(covariance, n_numrows, m_numcols, option):
        """The function is used to restructure the covariance Matrix
        differently for the update of `G` and `F` Matrices.

        Parameters
        ----------
        covariance : ndarray
            Size -> nm`x`nm\n
            The original covariance Matrix. Python flattens the `X` matrix by
            order 'C', i.e., Row elements are stacked one below another.
        n_numrows : float
            Number of numrows.
        m_numcols : float
            Number of numcols.
        option : ('row_stacked', 'column_stacked')
            Option to specify if the covariance matrix is obtained by
            row-stacking or column stacking the elements of the data matrix (X_matrix)

        Returns
        -------
        covariance_column_inverse_F_upd : ndarray
            Size -> nm`x`nm\n
            covariance matrix structured for the update of `F`.
        covariance_row_inverse_G_upd : ndarray
            Size -> nm`x`nm\n
            covariance matrix structured for the update of `G`.
        """

        # check_positive_Definite = internal_functions.is_pos_def(covariance)

        #if check_positive_Definite==0:
        #    raise Exception("Covariance Matrix is not Positive Definite")

        if option == 'row_stacked':
            # Row Stacked covariance Matrix -- Update of G
            covariance_row = covariance
            covariance_row_inverse_G_upd = np.array(np.linalg.inv(covariance_row))

            # Column Stacking covariance Matrix -- Update of F
            covariance_column = np.zeros(covariance_row.shape)
            indI = np.empty(0, dtype='int')
            for l in range(m_numcols):
                indI = np.append(indI, np.arange(0+l, (n_numrows*m_numcols)+l, m_numcols))

            indJ = indI
            for a, i in enumerate(indI):
                for b, j in enumerate(indJ):
                    covariance_column[a, b] = covariance_row[i, j]

            covariance_column_inverse_F_upd = np.array(np.linalg.inv(covariance_column))

        elif option == 'column_stacked':
            # Column Stacked covariance Matrix -- Update of F
            covariance_column = covariance
            covariance_column_inverse_F_upd = np.array(np.linalg.inv(covariance_column))

            # Row Stacking covariance Matrix -- Update of G
            covariance_row = np.zeros(covariance_column.shape)
            indI = np.empty(0, dtype='int')
            for l in range(n_numrows):
                indI = np.append(indI, np.arange(0+l, (n_numrows*m_numcols)+l, n_numrows))

            indJ = indI
            for a, i in enumerate(indI):
                for b, j in enumerate(indJ):
                    covariance_row[a, b] = covariance_column[i, j]

            covariance_row_inverse_G_upd = np.array(np.linalg.inv(covariance_row))

        else:
            raise Exception('Please mention if the covariance matrix is obtained by ``row_stacked`` the elements of X matrix or ``column_stacked`` the elements of X matrix.')

        return covariance_column_inverse_F_upd, covariance_row_inverse_G_upd

    ## Splitting the covariance Matrix ======================== ##
    def split_covariance_inverse(covariance, n_numrows, m_numcols, option = ("row_stacked", "column_stacked")):
        """This function is used to split the Inverse of the covariance Matrix
        into a plus part and a minus part. Refer to Plis et. al. (2011)[1] for
        more details.

        Parameters
        ----------
        covariance : ndarray
            Size -> nm`x`nm
            covariance Matrix.
        n_numrows : int
            Number of numrows.
        m_numcols : int
            Number of numcols.
        option : ('row_stacked', 'column_stacked')
            Option to specify if the covariance matrix is obtained by
            row-stacking or column stacking the elements of the data matrix (X_matrix)

        Returns
        -------
        SF_plus : ndarray
            Size -> nm`x`nm
            plus Part of covariance Matrix for the update of F.
        SF_minus : ndarray
            Size -> nm`x`nm
            minus Part of covariance Matrix for the update of F.
        SG_plus : ndarray
            Size -> nm`x`nm
            plus Part of covariance Matrix for the update of G.
        SG_minus : ndarray
            Size -> nm`x`nm
            minus Part of covariance Matrix for the update of G.

        References
        ----------
        [1] Plis, S. M., Potluru, V. K., Lane, T., & Calhoun, V. D. (2011).
        Correlated noise: How it breaks NMF, and what to do about it.
        Journal of signal processing systems, 65(3), 351-359.
        <https://link.springer.com/article/10.1007/s11265-010-0511-8>
        """
        # Getting the inverse of matrices for the update of F and G
        S_F_upd, S_G_upd = covariance_matrix_handling.restructure_covariance_inverse(covariance, n_numrows, m_numcols, option=option)

        # Generating intermediate blank split matrices for the update of F and G
        SF_INT_plus, SF_INT_minus = np.zeros(S_F_upd.shape), np.zeros(S_F_upd.shape)
        SG_INT_plus, SG_INT_minus = np.zeros(S_G_upd.shape), np.zeros(S_G_upd.shape)

        # Populating blank split matrices
        for i in range(S_F_upd.shape[0]):
            for j in range(S_F_upd.shape[1]):
                if S_F_upd[i, j] >= 0:
                    SF_INT_plus[i, j] = S_F_upd[i, j]
                    SF_INT_minus[i, j] = 0
                elif S_F_upd[i, j] < 0:
                    SF_INT_plus[i, j] = 0
                    SF_INT_minus[i, j] = np.abs(S_F_upd[i, j])

                if S_G_upd[i, j] >= 0:
                    SG_INT_plus[i, j] = S_G_upd[i, j]
                    SG_INT_minus[i, j] = 0
                elif S_G_upd[i, j] < 0:
                    SG_INT_plus[i, j] = 0
                    SG_INT_minus[i, j] = np.abs(S_G_upd[i, j])

        # Determining the minimum eigen values for minus part of the Matrix.
        SF_minus_eig = np.linalg.eigh(SF_INT_minus)[0]
        SF_plus_eig = np.linalg.eigh(SF_INT_plus)[0]
        SG_minus_eig = np.linalg.eigh(SG_INT_minus)[0]
        SG_plus_eig = np.linalg.eigh(SG_INT_plus)[0]



        min_neg_eigen_F_upd = min(np.append(SF_minus_eig, SF_plus_eig))
        min_neg_eigen_G_upd = min(np.append(SG_minus_eig, SG_plus_eig))

        # Generating final split matrices for the update of F and G
        if min_neg_eigen_F_upd > 0:
            min_neg_eigen_F_upd = 0
            SF_plus = SF_INT_plus
            SF_minus = SF_INT_minus
        else:
            min_neg_eig = np.abs(min_neg_eigen_F_upd)
            SF_plus = SF_INT_plus + min_neg_eig*np.eye(SF_INT_minus.shape[0])
            SF_minus = SF_INT_minus + min_neg_eig*np.eye(SF_INT_minus.shape[0])

        if min_neg_eigen_G_upd > 0:
            min_neg_eigen_F_upd = 0
            SF_plus = SF_INT_plus
            SF_minus = SF_INT_minus
        else:
            min_neg_eig = np.abs(min_neg_eigen_G_upd)
            SG_plus = SG_INT_plus + min_neg_eig*np.eye(SG_INT_minus.shape[0])
            SG_minus = SG_INT_minus + min_neg_eig*np.eye(SG_INT_minus.shape[0])

        if option == 'row_stacked':
            return SF_plus, SF_minus, SG_plus, SG_minus
        elif option == 'column_stacked':
            return SG_plus, SG_minus, SF_plus, SF_minus

    ## ====================================================================== ##

##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
class gnmf_projected_gradient:
    """This class is used to code the proposed Generalised Non-Negative
    Matrix Factorisation with Projected Gradient Approach

    Description
    -----------
    Generalised Non-Negative Matrix Factorisation Method (GNMF) method
    contains several functions,

    `gnmf_update_F` : function
        This function is used to update the value of `F`.

    `alpha_selection_F` : function
        This function is used to selected the method for step-length selection.

    `gnmf_update_G` : function
        This function is used to update the value of `G`.

    `alpha_selection_G` : function
        This function is used to selected the method for step-length selection.

    `objective_function` : function
        This function is used to compute the objective function of an iteration
        (after updating both `G` and `F` matrices). This is the objective
        function we wish to minimise and the whole formulation is structured
        around this objective function.

    `running_method` : function
        This function is used to run the GNMF-PG method with
    """

    ## GNMF Method --> Update of F ========================================== ##
    def gnmf_update_F(X_matrix, G_init, F_init, covariance_inverse, SX_G_upd, alpha_init, beta, sigma):
        """This function is used for updating `F` using the GNMF problem.

        Parameters
        ----------
        X_matrix : ndarray
            Size -> n`x`m\n
            Input matrix with dimension (n`x`m)
        G_init : ndarray
            Size -> n`x`k\n
            Source contribution matrix of size (n`x`k)
        F_init : ndarray
            Size -> k`x`m\n
            Source profile matrix of size (n`x`k)
        covariance_inverse : ndarray
            Size -> nm`x`nm\n
            covariance matrix of size (nm`x`nm)
        alpha_init : float
            Initial Alpha value
        beta : float
            Beta value
        sigma : float
            Sigma float

        Returns
        -------
        F_upd : ndarray
            Size -> n`x`k\n
            Updated value of F.
        alpha_value : float
            Step-length Value. The value is obtained by Lin's Method.
        OFunc_value : float
            Objective Function Value
        """
        # Checking the Dimensions
        n, m = X_matrix.shape
        r, k= G_init.shape
        k, u = F_init.shape

        if (n != r and m != u):
            print('Check the dimensions')
        else:
            # Arranging the matrices
            X_c = X_matrix.T.flatten()
            F_k = F_init.T.flatten()
            G_mat = np.kron(np.eye(m), G_init)

            # Calculating grad_F matrix
            grad_F = np.empty(0)
            ScGmatFvec = covariance_inverse@((G_init@F_init).T.flatten())
            for i in range(m):
                grad_F = np.append(grad_F, -1*G_init.T@SX_G_upd[n*i:n*(i+1)] + (G_init.T@ScGmatFvec[n*i:n*(i+1)]))

            ## Computing the updated F
            F_kp1 = F_k - alpha_init*grad_F

            ## Checking for negative values and setting them as zero.
            F_kp1[F_kp1 < 0] = 0

            Xc_minus_Gmat_Fk = X_c - (G_init@F_k.reshape(m, k).T).T.flatten()
            Xc_minus_Gmat_Fkp1 = X_c - (G_init@F_kp1.reshape(m, k).T).T.flatten()
            obj_fun_oldF = 0.5*((Xc_minus_Gmat_Fk).T@covariance_inverse)@(Xc_minus_Gmat_Fk)
            obj_fun_newF = 0.5*((Xc_minus_Gmat_Fkp1).T@covariance_inverse)@(Xc_minus_Gmat_Fkp1)
            it_val = 0
            alpha_F = alpha_init
            while (obj_fun_newF - obj_fun_oldF > sigma*grad_F.T@(F_kp1 - F_k)) :
                alpha_F = alpha_F * beta
                F_kp1 = F_k - alpha_F*grad_F
                F_kp1[F_kp1 < 0] = 0
                Xc_minus_Gmat_Fkp1 = X_c - (G_init@F_kp1.reshape(m, k).T).T.flatten()
                obj_fun_newF = 0.5*((Xc_minus_Gmat_Fkp1).T@covariance_inverse)@(Xc_minus_Gmat_Fkp1)
                it_val = it_val + 1
                alpha_init = alpha_F

        return F_kp1.reshape(u, k).T, alpha_init, obj_fun_newF
    ## ====================================================================== ##

    ## GNMF Method --> Update of G ========================================== ##
    def gnmf_update_G(X_matrix, G_init, F_init, covariance_inverse, SX_G_upd, alpha_init, beta, sigma):
        """This function is used for updating `F` using the GNMF problem.

        Parameters
        ----------
        X : ndarray
            Size -> n`x`m\n
            Input matrix with dimension (n`x`m)
        G0 : ndarray
            Size -> n`x`k\n
            Source contribution matrix of size (n`x`k)
        F0 : ndarray
            Size -> k`x`m\n
            Source profile matrix of size (n`x`k)
        covariance_inverse : ndarray
            Size -> nm`x`nm\n
            covariance matrix of size (nm`x`nm)
        alpha_init : float
            Initial Alpha value
        beta : float
            Beta value
        sigma : float
            Sigma float

        Returns
        -------
        G_new : ndarray
            Size -> n`x`k\n
            Updated value of G.
        alpha_val : float
            Step-length Value. The value is obtained by Lin's Method.
        objfun_value : float
            Objective Function Value
        """
        n, m = X_matrix.shape
        r, k = G_init.shape
        k, u = F_init.shape

        if (n != r and m != u):
            raise Exception("The dimensions of the input matrices is incorrect.")
        else:
            X_r = X_matrix.flatten()
            G_k = G_init.flatten()
            Sr_F_matT_G_vecT = covariance_inverse@(F_init.T@G_init.T).T.flatten()
            grad_G = np.empty(0)
            for i in range(n):
                grad_G = np.append(grad_G, -1*(F_init@SX_G_upd[m*i:m*(i+1)]) + (F_init@Sr_F_matT_G_vecT[m*i:m* (i+1)]))

            G_kp1 = G_k - alpha_init*grad_G
            G_kp1[G_kp1 < 0] = 0
            Xr_minus_G_k_F_mat = X_r - (G_k.reshape(n, k)@F_init).flatten()
            Xr_minus_G_kp1_F_mat = X_r - (G_kp1.reshape(n, k)@F_init).flatten()
            obj_fun_oldG = 0.5*((Xr_minus_G_k_F_mat).T)@covariance_inverse@((Xr_minus_G_k_F_mat))
            obj_fun_newG = 0.5*((Xr_minus_G_kp1_F_mat).T)@covariance_inverse@((Xr_minus_G_kp1_F_mat))
            it_val = 0
            alpha_G = alpha_init
            while ((obj_fun_newG-obj_fun_oldG) > (sigma*grad_G.T@(G_kp1-G_k))):
                alpha_G = alpha_G*beta
                G_kp1 = G_k - alpha_G*grad_G
                G_kp1[G_kp1 < 0] = 0
                Xr_minus_G_kp1_F_mat = X_r - (G_kp1.reshape(n, k)@F_init).flatten()
                obj_fun_newG = 0.5*((Xr_minus_G_kp1_F_mat).T)@covariance_inverse@((Xr_minus_G_kp1_F_mat))
                it_val = it_val + 1
                alpha_init = alpha_G

            G_vec_upd = G_k - alpha_G*grad_G
            G_vec_upd[G_vec_upd < 0] = 0
            G_upd = G_vec_upd.reshape(n, k)

        return G_upd, alpha_init, obj_fun_newG
    ## ====================================================================== ##

    ## GNMF Method --> Objective Function ==================================== ##
    def objective_function(X_matrix, G_init, F_init, covariance_inverse, option):
        """The function returns the Objective Function value after the update
        of both `G` and `F` Matrices

        Parameters
        ----------
        X_matrix : ndarray
            Size -> n`x`m\n
            Input matrix
        G_init : ndarray
            Size -> n`x`k\n
            Source contribution matrix
        F_init : ndarray
            Size -> k`x`m\n
            Source profile matrix
        covariance_inverse : ndarray
            Size -> nm`x`nm\n
            covariance Matrix

        Returns
        -------
        obj_func : float
            Objective Function Values.
        """
        if option == 'row_stacked':
            unit = (X_matrix - G_init@F_init).flatten()
            obj_func = 0.5*(unit@covariance_inverse@unit)
        elif option == 'column_stacked':
            unit = (X_matrix - G_init@F_init).T.flatten()
            obj_func = 0.5*(unit@covariance_inverse@unit)

        return obj_func

    def running_method(X_matrix,
                       covariance,
                       G_init = ('random'),
                       F_init = ('random'),
                       beta = 0.1,
                       sigma = 0.0001,
                       alpha_init_G = 1,
                       alpha_init_F = 1,
                       option = ('row_stacked', 'column_stacked'),
                       num_fact=None,
                       num_init = 1,
                       max_iter = 500000,
                       tolerance = 1e-6,
                       conv_typ = 'relative',
                       conv_num = 3):
        """The function return runs the projected gradient method under consideration.

        Parameters
        ----------
        X_matrix : ndarray
            Size -> n`x`m\n
            Input Matrix.
        covariance : ndarray
            Size -> nm`x`nm\n
            covariance Matrix.
        G_init : ndarray
            Size -> n`x`k\n
            Initial Source Contribution Matrix.
        F_init : ndarray
            Size -> k`x`m\n
            Initial Source Profile Matrix.
        beta : float
            Number by which the alpha value reduces when to satisfy the
            sufficient decrease condition. Default value = 0.1
        sigma : float
            Parameter for the sufficient decrease condition. Default Value = 0.0001
        alpha_init_G : float
            Initial value for the step-length for the update of G. Default Value = 1
        alpha_init_F : float
            Initial value for the step-length for the update of F. Default Value = 1
        option : ('row_stacked', 'column_stacked')
            Option to specify if the covariance matrix is obtained by
            row-stacking or column stacking the elements of the data matrix (X_matrix)
        num_fact : int
            Total Number of Factors
        num_init : int
            Total Number of initialisations for the dataset under consideration.
            Default = 1
        max_ter : int
            Total Number of allowable iterations. Default = 500000
        tolerance : float
            Tolerance value below which the method is considered converged.
            Default = 1e-6
        conv_typ : option
            Type of convergence i.e., should the absolute difference or the
            relative deviation in the objective values to be considered.
            Default = 'relative'
        conv_num : float
            Number of consecutive iteration for which the tolerance criteria
            should be met. Default = 10

        Returns
        -------
        G_upd : ndarray
            Source Contribution Matrix of size num_init`x`n`x`k.
        F_upd : ndarray
            Source Contribution Matrix of size num_init`x`k`x`m.
        obj_func : ndarray
            Iteration-wise Objective Function value of size num_init`x`max_iter

        """

        n_numrows, m_numcols = X_matrix.shape
        p_numfact = num_fact
        G_init, F_init = internal_functions.checking_initial_guess(G_init, F_init, n_numrows, m_numcols, p_numfact, num_init)

        ## Checking inputs -- Convergence
        if conv_typ == 'relative':
            def convergence_checking(OFunc_km1, OFunc_k):
                return np.abs((OFunc_km1 - OFunc_k)/OFunc_km1)
        elif conv_typ == 'absolute':
            def convergence_checking(OFunc_km1, OFunc_k):
                return np.abs(OFunc_km1 - OFunc_k)
        else:
            raise Exception("Convergence type required. Choose between 'relative' and 'absolute'.")

        print("Following are the Parameters Selected:\n======================================\nnumrows: \t\t {0},\nnumcols: \t\t {1},\nFactors: \t\t {2},\nConv. Type: \t\t {3},\nTolerance: \t\t less than {4} for {5} iterations,\nMax. Iter: \t\t {6}".format(n_numrows, m_numcols, p_numfact, conv_typ, tolerance, conv_num, max_iter))

        ## Preparing for the run -- Getting the covariance Matrix Sorted
        if option == 'row_stacked':
            covariance_inverse_F_upd, covariance_inverse_G_upd = covariance_matrix_handling.restructure_covariance_inverse(covariance,
                n_numrows = n_numrows,
                m_numcols = m_numcols,
                option = 'row_stacked'
            )
        elif option == 'column_stacked':
            covariance_inverse_F_upd, covariance_inverse_G_upd = covariance_matrix_handling.restructure_covariance_inverse(
                covariance,
                n_numrows = n_numrows,
                m_numcols = m_numcols,
                option = 'column_stacked'
            )
        #else:
            #raise ValueError:
            #continue

        ## Preparing for run -- Initialising
        obj_func = np.zeros((num_init, max_iter+1))
        G_mat = np.zeros((num_init, n_numrows, p_numfact))
        F_mat = np.zeros((num_init, p_numfact, m_numcols))

        for i in range(num_init):
            ## Preparing for run -- Initialising
            it = 0 ## Initialising the number of iterations
            delta = 10*[False] ## Initialising the delta as difference between the objective function values

            ## Starting the run
            G_run = G_init[i]
            F_run = F_init[i]
            obj_func_internal = np.zeros(max_iter+1)
            obj_func_internal[it] = gnmf_projected_gradient.objective_function(X_matrix, G_run, F_run, covariance_inverse_G_upd, option = option)

            ## CONSTANT TERMS
            Xc = X_matrix.T.flatten()
            SX_F_upd = covariance_inverse_F_upd@Xc

            ## SrXr Calculation
            Xr = X_matrix.flatten()
            SX_G_upd = covariance_inverse_G_upd@Xr

            check_convergence = True
            with Progress() as progress:
                task1 = progress.add_task("[red] GNMF-MU", total=max_iter)
                while (it < max_iter) and check_convergence:

                    it = it + 1

                    ## Update F Matrix
                    F_upd, alphaF, _ = gnmf_projected_gradient.gnmf_update_F(X_matrix, G_run, F_run, covariance_inverse_F_upd, SX_F_upd, alpha_init=alpha_init_F, beta = beta, sigma = sigma)

                    ## Update G Matrix
                    G_upd, alphaG, ofunc_value_from_G_update = gnmf_projected_gradient.gnmf_update_G(X_matrix, G_run, F_upd, covariance_inverse_G_upd, SX_G_upd, alpha_init=alpha_init_G, beta = beta, sigma = sigma)

                    ## Objective Function
                    obj_func_internal[it] = ofunc_value_from_G_update

                    ## Since Objective Function is computed as part for the update
                    # of G and F Matrices, the step is commented.
                    #gnmf_projected_gradient.objective_function(X_matrix, G_upd,
                    # F_upd, covariance_inverse_G_upd, option = option)

                    ## Convergence criteria calculation
                    conv_criteria = float(convergence_checking(obj_func_internal[it-1], obj_func_internal[it]))
                    #print(conv_criteria)
                    delta.append(conv_criteria < tolerance)

                    ## Check for convergence in terms of difference in the objective function
                    check_convergence =  sum(delta[it-conv_num::]) < conv_num

                    F_run, G_run = F_upd, G_upd

                    progress.update(task1, advance = 1, description="[bold yellow]GNMF Projected Gradient:\nδ: \t{}, \nJ: \t{}, \nα_G: \t{}, \nα_F: \t{}, \nit: \t{}/{}".format("{:.4e}".format(conv_criteria), "{:.4e}".format((float(obj_func_internal[it]))), "{:.4e}".format(alphaG), "{:.4e}".format(alphaF), it, max_iter ))

            G_mat[i, :, :] = G_upd
            F_mat[i, :, :] = F_upd
            obj_func[i, :] = obj_func_internal

        return G_mat, F_mat, obj_func

##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
class gnmf_multiplicative_update:
    """This class is used to implement GNMF Method with
    multiplicative updates.

    Description
    -----------
    Generalised Non-Negative Matrix Factorisation (GNMF) is described in this
    class.  All the variants of the multiplicative methods including glsNMF with
    correlated rows or correlated columns, LS-NMF with diagonal covariance matrix
    and NMF method described by Lee and Seung with diagonal covariance matrix.

    Following are some functions which are part
    of the class,

    `update_F` : function
        This function is used for the update of F.
    `update_G` : function
        This function is used for the update of G.
    `objective_function` : function
        This function is used to compute the value of objective function
    `running_method` : function
        This function is used to run the method under consideration.

    """

    def gnmf_update_G(X_matrix,
                      G_init,
                      F_init,
                      SG_plus,
                      SG_minus,
                      SG_plus_X_vec_G,
                      SG_minus_X_vec_G):
        """This function is used for the update of `G` Matrix using the
        Multiplicative Method discussed in Plis et. al. (2011).

        Parameters
        ----------
        X : ndarray
            Size -> n`x`m\n
            Input matrix
        G_init : ndarray
            Size -> n`x`k\n
            Initial Value of Source Contribution Matrix.
        F_init : ndarray
            Size -> k`x`m\n
            Initial Value of Source Profile Matrix.
        SG_plus : ndarray
            Size -> nm`x`nm
            Split of covariance Matrix with all the positive values and zero,
            for the update of `G` Matrix. Details about how the split is done
            is given in Plis et. al. (2011).
        SG_minus : ndarray
            Size -> nm`x`nm
            Split of covariance Matrix with all the negative values, for the
            update of `G` Matrix. Details about how the split is done is given
            in Plis et. al. (2011).

        Returns
        -------
        G_upd : ndarray
            size -> n`x`k\n
            Updated value of Source Contribution Matrix.

        References
        ----------
        [1] Plis, S. M., Potluru, V. K., Lane, T., & Calhoun, V. D. (2011).
        Correlated noise: How it breaks NMF, and what to do about it.
        Journal of signal processing systems, 65(3), 351-359.
        <https://link.springer.com/article/10.1007/s11265-010-0511-8>
        """
        # Checking the dimension of the problem
        n_numrows, m_numcols = X_matrix.shape
        kf, p_numfact = G_init.shape
        u, kg = F_init.shape

        if (n_numrows != kf and m_numcols != kg and p_numfact != u):
            print('Check the Dimensions')
        else:
            F_mat = np.kron(np.eye(n_numrows), F_init)
            G_initF_init = (G_init@F_init).flatten()


            # Update of G -- Numerator
            G_upd_num = F_mat@(SG_plus_X_vec_G + SG_minus@G_initF_init)
            G_upd_den = F_mat@(SG_minus_X_vec_G + SG_plus@G_initF_init)

            G_upd = np.multiply(G_init, np.divide(G_upd_num, G_upd_den).reshape(n_numrows, p_numfact))

        return G_upd

    def gnmf_update_F(X_matrix,
                      G_init,
                      F_init,
                      SF_plus,
                      SF_minus,
                      SF_plus_X_vec_F,
                      SF_minus_X_vec_F):
        """This function is used for the update of `G` Matrix using the
        Multiplicative Method discussed in Plis et. al. (2011).

        Parameters
        ----------
        X : ndarray
            Size -> n`x`m\n
            Input matrix
        G_init : ndarray
            Size -> n`x`k\n
            Initial Value of Source Contribution Matrix.
        F_init : ndarray
            Size -> k`x`m\n
            Initial Value of Source Profile Matrix.
        SF_plus : ndarray
            Size -> nm`x`nm
            Split of covariance Matrix with all the positive values and zero,
            for the update of `F` Matrix. Details about how the split is done
            is given in Plis et. al. (2011).
        SF_minus : ndarray
            Size -> nm`x`nm
            Split of covariance Matrix with all the negative values, for the
            update of `F` Matrix. Details about how the split is done is given
            in Plis et. al. (2011).
        SF_plus_X_vec_F : ndarray
            Precomputed the constant matrix obtained by multiplying SF_plus with
            appropriate X Matrix.
        SF_minus_X_vec_F : ndarray
            Precomputed the constant matrix obtained by multiplying SF_minus with
            appropriate X Matrix.
        Returns
        -------
        F_upd : ndarray
            size -> k`x`m\n
            Updated value of Source Profile Matrix.

        References
        ----------
        [1] Plis, S. M., Potluru, V. K., Lane, T., & Calhoun, V. D. (2011).
        Correlated noise: How it breaks NMF, and what to do about it.
        Journal of signal processing systems, 65(3), 351-359.
        <https://link.springer.com/article/10.1007/s11265-010-0511-8>
        """

        # Checking the dimension of the problem
        n_numrows, m_numcols = X_matrix.shape
        r, p_numfact = G_init.shape
        p_numfact, u = F_init.shape

        if (n_numrows != r and m_numcols != u):
            print('Check the Dimensions')
        else:
            G_mat = np.kron(np.eye(m_numcols), G_init)
            G_initF_init = (G_init@F_init).T.flatten()


            # Update of G -- Numerator
            F_upd_num = G_mat.T@(SF_plus_X_vec_F + SF_minus@G_initF_init)
            F_upd_den = G_mat.T@(SF_minus_X_vec_F + SF_plus@G_initF_init)

            F_upd = np.multiply(F_init, np.divide(F_upd_num, F_upd_den).reshape(m_numcols, p_numfact).T)

        return F_upd

    def objective_function(X_matrix,
                           G_upd,
                           F_upd,
                           covariance_inverse,
                           option = ('row_stacked', 'column_stacked')):
        """The function return the Objective Function value after the
        update of both `G` and `F` Matrices.

        Parameters
        ----------
        X : ndarray
            Size -> n`x`m\n
            Input Matrix.
        G_upd : ndarray
            Size -> n`x`k\n
            Source Contribution Matrix.
        F_upd : ndarray
            Size -> k`x`m\n
            Source Profile Matrix.
        covariance_inverse : ndarray
            Size -> nm`x`nm\n
            covariance Matrix.
        option : ('row_stacked', 'column_stacked')
            Option to specify if the covariance matrix is obtained by
            row-stacking or column stacking the elements of the data matrix (X_matrix)


        Returns
        -------
        obj_func : float
            Objective function value.

        Notes
        -----
        The Objective Function contains the Inverse of covariance Matrix
        because we want to see how the method performs when the Objective
        Function of the problem is changed.
        """
        if option == 'row_stacked':
            unit = (X_matrix - G_upd@F_upd).flatten()
            ofunc_value = 0.5*(unit@covariance_inverse@unit)
        elif option == 'column_stacked':
            unit = (X_matrix - G_upd@F_upd).T.flatten()
            ofunc_value = 0.5*(unit@covariance_inverse@unit)

        return ofunc_value


    def running_method(X_matrix,
                       covariance,
                       option = ('row_stacked', 'column_stacked'),
                       G_init = 'random',
                       F_init = 'random',
                       num_fact=None,
                       num_init = 1,
                       max_iter = 500000,
                       tolerance = 1e-6,
                       conv_typ = 'relative',
                       conv_num = 3):
        """The function return runs the multiplicative method under consideration.

        Parameters
        ----------
        X_matrix : ndarray
            Size -> n`x`m\n
            Input Matrix.
        covariance : ndarray
            Size -> nm`x`nm\n
            covariance Matrix.
        G_init : ndarray
            Size -> n`x`k\n
            Initial Source Contribution Matrix.
        F_init : ndarray
            Size -> k`x`m\n
            Initial Source Profile Matrix.
        num_fact : int
            Total Number of Factors
        num_init : int
            Total Number of initialisations for the dataset under consideration.
            Default = 1
        max_ter : int
            Total Number of allowable iterations. Default = 500000
        tolerance : float
            Tolerance value below which the method is considered converged.
            Default = 1e-6
        conv_typ : option
            Type of convergence i.e., should the absolute difference or the
            relative deviation in the objective values to be considered.
            Default = 'relative'
        conv_num : float
            Number of consecutive iteration for which the tolerance criteria
            should be met. Default = 10

        Returns
        -------
        G_upd : ndarray
            Source Contribution Matrix of size num_init`x`n`x`k.
        F_upd : ndarray
            Source Contribution Matrix of size num_init`x`k`x`m.
        obj_func : ndarray
            Iteration-wise Objective Function value of size num_init`x`max_iter

        Notes
        -----
        The Objective Function contains the Inverse of covariance Matrix
        because we want to see how the method performs when the Objective
        Function of the problem is changed.
        """

        ## Performing checks on initial guess
        n_numrows, m_numcols = X_matrix.shape
        p_numfact = num_fact
        G_init, F_init = internal_functions.checking_initial_guess(G_init, F_init, n_numrows, m_numcols, p_numfact, num_init)

        ## Checking convergence options
        if conv_typ == 'relative':
            def convergence_checking(OFunc_km1, OFunc_k):
                return np.abs((OFunc_km1 - OFunc_k)/OFunc_km1)
        elif conv_typ == 'absolute':
            def convergence_checking(OFunc_km1, OFunc_k):
                return np.abs(OFunc_km1 - OFunc_k)
        else:
            raise Exception("Convergence type required. Choose between 'relative' and 'absolute'.")

        print("Following are the Parameters Selected:\n======================================\nnumrows: \t\t {0},\nnumcols: \t\t {1},\nFactors: \t\t {2},\nConv. Type: \t\t {3},\nTolerance: \t\t less than {4} for {5} iterations,\nMax. Iter: \t\t {6}".format(n_numrows, m_numcols, p_numfact, conv_typ, tolerance, conv_num, max_iter))

        ## Splitting the matrix based on the option provided
        if option == 'row_stacked':
            SF_plus, SF_minus, SG_plus, SG_minus = covariance_matrix_handling.split_covariance_inverse(
                covariance,
                n_numrows,
                m_numcols,
                option = 'row_stacked'
            )

            ## Preparing for the run -- Split Matrices
            SF_plus_X_vec_F = SF_plus@X_matrix.T.flatten()
            SF_minus_X_vec_F = SF_minus@X_matrix.T.flatten()
            SG_plus_X_vec_G = SG_plus@X_matrix.flatten()
            SG_minus_X_vec_G = SG_minus@X_matrix.flatten()
            covariance_inverse = np.array(np.linalg.inv(covariance))
            if np.linalg.norm((SG_plus - SG_minus) - covariance_inverse) > 1e-3:
                print(np.linalg.norm((SG_plus - SG_minus) - covariance_inverse))
                raise Exception("The split operation is not performed properly")
        elif option == 'column_stacked':
            SG_plus, SG_minus, SF_plus, SF_minus = covariance_matrix_handling.split_covariance_inverse(
                covariance,
                n_numrows,
                m_numcols,
                option = 'column_stacked'
            )

            ## Preparing for the run -- Split Matrices
            SF_plus_X_vec_F = SF_plus@X_matrix.flatten()
            SF_minus_X_vec_F = SF_minus@X_matrix.flatten()
            SG_plus_X_vec_G = SG_plus@X_matrix.T.flatten()
            SG_minus_X_vec_G = SG_minus@X_matrix.T.flatten()
            covariance_inverse = np.array(np.linalg.inv(covariance))
            if np.linalg.norm((SF_plus - SF_minus) - covariance_inverse) > 1e-3:
                print(np.linalg.norm((SF_plus - SF_minus) - covariance_inverse))
                raise Exception("The split operation is not performed properly")

        ## Preparing for run -- Initialising G, F and Objective Function
        obj_func = np.zeros((num_init, max_iter+1))
        G_mat = np.zeros((num_init, n_numrows, p_numfact))
        F_mat = np.zeros((num_init, p_numfact, m_numcols))

        for i in range(num_init):

            ## Preparing for run -- Initialising
            it = 0 ## Initialising the number of iterations
            delta = 10*[False] ## Initialising the delta as difference between the objective function values

            ## Starting the run
            G_run = G_init[i]
            F_run = F_init[i]
            obj_func_internal = np.zeros(max_iter+1)
            obj_func_internal[it] = gnmf_multiplicative_update.objective_function(X_matrix, G_run, F_run, covariance_inverse, option = option)

            check_convergence = True
            with Progress() as progress:
                task1 = progress.add_task("[red] GNMF Multiplicative Update", total=max_iter)
                while (it < max_iter) and check_convergence:
                    #print((np.sum(np.array(delta[it:it+10]) < tolerance) == conv_num))
                    it = it + 1

                    ## Update F Matrix
                    F_upd = gnmf_multiplicative_update.gnmf_update_F(X_matrix, G_run, F_run, SF_plus, SF_minus, SF_plus_X_vec_F, SF_minus_X_vec_F)

                    ## Update G Matrix
                    G_upd = gnmf_multiplicative_update.gnmf_update_G(X_matrix, G_run, F_upd, SG_plus, SG_minus, SG_plus_X_vec_G, SG_minus_X_vec_G)

                    ## Objective Function
                    obj_func_internal[it] = gnmf_multiplicative_update.objective_function(X_matrix, G_upd, F_upd, covariance_inverse, option = option)

                    ## Convergence criteria calculation
                    conv_criteria = float(convergence_checking(obj_func_internal[it-1], obj_func_internal[it]))
                    delta.append(conv_criteria < tolerance)

                    ## Check for convergence in terms of difference in the objective function
                    check_convergence =  sum(delta[it-conv_num::]) < conv_num



                    F_run, G_run = F_upd, G_upd
                    progress.update(task1, advance = 1, description="[bold green]GNMF Multiplicative Update:\nδ: \t{}, \nJ: \t{}, \nit: \t{}/{}".format("{:.4e}".format(conv_criteria), "{:.4e}".format(float(obj_func_internal[it])), it, max_iter))

            #tqdm.close(pbar) # Closing the tqdm bar
            G_mat[i, :, :] = G_upd
            F_mat[i, :, :] = F_upd
            obj_func[i, :] = obj_func_internal

        return G_mat, F_mat, obj_func

##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇
##▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇

class nmf_multiplicative_update:
    """This class is used to implement NNMF Method introduced by Lee and Seung
    with multiplicative updates.

    Description
    -----------
    Following are some functions which are part
    of the class,

    `objective_function` : function
        This function is used to compute the value of objective function
    `running_method` : function
        This function is used to run the method under consideration.
    """

    def objective_function(X_matrix, G_upd, F_upd):
        """The function return the Objective Function value after the
        update of both `G` and `F` Matrices.

        Parameters
        ----------
        X : ndarray
            Size -> n`x`m\n
            Input Matrix.
        G0 : ndarray
            Size -> n`x`k\n
            Source Contribution Matrix.
        F0 : ndarray
            Size -> k`x`m\n
            Source Profile Matrix.
        covariance_inverse : ndarray
            Size -> nm`x`nm\n
            covariance Matrix.

        Returns
        -------
        obj_func : float
            Objective function value.

        Notes
        -----
        The Objective Function contains the Inverse of covariance Matrix
        because we want to see how the method performs when the Objective
        Function of the problem is changed.
        """
        unit = (X_matrix - G_upd@F_upd)
        ofunc_value = 0.5*np.sum(np.trace(unit.T@unit))

        return ofunc_value


    def running_method(X_matrix,
                       G_init = ('random'),
                       F_init = ('random'),
                       num_fact=None,
                       num_init = 1,
                       max_iter = 500000,
                       tolerance = 1e-6,
                       conv_typ = 'relative',
                       conv_num = 3):
        """The function return runs the projected gradient method under consideration.

        Parameters
        ----------
        X_matrix : ndarray
            Size -> n`x`m\n
            Input Matrix.
        G_init : ndarray
            Size -> n`x`k\n
            Initial Source Contribution Matrix.
        F_init : ndarray
            Size -> k`x`m\n
            Initial Source Profile Matrix.
        num_fact : int
            Total Number of Factors
        num_init : int
            Total Number of initialisations for the dataset under consideration.
            Default = 1
        max_ter : int
            Total Number of allowable iterations. Default = 500000
        tolerance : float
            Tolerance value below which the method is considered converged.
            Default = 1e-6
        conv_typ : option
            Type of convergence i.e., should the absolute difference or the
            relative deviation in the objective values to be considered.
            Default = 'relative'
        conv_num : float
            Number of consecutive iteration for which the tolerance criteria
            should be met. Default = 10

        Returns
        -------
        G_upd : ndarray
            Source Contribution Matrix of size num_init`x`n`x`k.
        F_upd : ndarray
            Source Contribution Matrix of size num_init`x`k`x`m.
        obj_func : ndarray
            Iteration-wise Objective Function value of size num_init`x`max_iter

        """

        ## Performing checks on initial guess
        n_numrows, m_numcols = X_matrix.shape
        p_numfact = num_fact
        G_init, F_init = internal_functions.checking_initial_guess(G_init, F_init, n_numrows, m_numcols, p_numfact, num_init)

        ## Checking convergence options
        if conv_typ == 'relative':
            def convergence_checking(OFunc_km1, OFunc_k):
                return np.abs((OFunc_km1 - OFunc_k)/OFunc_km1)
        elif conv_typ == 'absolute':
            def convergence_checking(OFunc_km1, OFunc_k):
                return np.abs(OFunc_km1 - OFunc_k)
        else:
            raise Exception("Convergence type required. Choose between 'relative' and 'absolute'.")

        print("Following are the Parameters Selected:\n======================================\nnumrows: \t\t {0},\nnumcols: \t\t {1},\nFactors: \t\t {2},\nConv. Type: \t\t {3},\nTolerance: \t\t less than {4} for {5} iterations,\nMax. Iter: \t\t {6}".format(n_numrows, m_numcols, p_numfact, conv_typ, tolerance, conv_num, max_iter))

        ## Preparing for run -- Initialising
        obj_func = np.zeros((num_init, max_iter+1))
        G_mat = np.zeros((num_init, n_numrows, p_numfact))
        F_mat = np.zeros((num_init, p_numfact, m_numcols))

        for i in range(num_init):

            ## Preparing for run -- Initialising
            it = 0 ## Initialising the number of iterations
            delta = 10*[False] ## Initialising the delta as difference between the objective function values

            ## Starting the run
            G_run = G_init[i]
            F_run = F_init[i]
            obj_func_internal = np.zeros(max_iter+1)
            obj_func_internal[it] = nmf_multiplicative_update.objective_function(X_matrix, G_run, F_run)

            check_convergence = True
            with Progress() as progress:
                task1 = progress.add_task("[red] GNMF-MU", total=max_iter)
                while (it < max_iter) and check_convergence:
                    it = it + 1

                    ## Update F Matrix
                    F_upd = np.multiply(F_run, np.divide(G_run.T @ X_matrix, G_run.T @ (G_run @F_run)))

                    ## Update G Matrix
                    G_upd = np.multiply(G_run, np.divide(X_matrix @ F_upd.T, (G_run @ F_upd) @ F_upd.T))

                    ## Objective Function
                    obj_func_internal[it] = nmf_multiplicative_update.objective_function(X_matrix, G_upd, F_upd)

                    ## Convergence criteria calculation
                    conv_criteria = float(convergence_checking(obj_func_internal[it-1], obj_func_internal[it]))
                    delta.append(conv_criteria < tolerance)

                    ## Check for convergence in terms of difference in the objective function
                    check_convergence =  sum(delta[it-conv_num::]) < conv_num


                    F_run, G_run = F_upd, G_upd
                    progress.update(task1, advance = 1, description="[bold blue]NMF Multiplicative Update:\nδ: \t{}, \nJ: \t{}, \nit: \t{}/{}".format("{:.4e}".format(conv_criteria), "{:.4e}".format(float(obj_func_internal[it])), it, max_iter))

            G_mat[i, :, :] = G_upd
            F_mat[i, :, :] = F_upd
            obj_func[i, :] = obj_func_internal

        return G_mat, F_mat, obj_func
