# pyGNMF
 Python Library for Generalised Non-Negative Matrix Factorisiation (GNMF)

This Python library implements GNMF method introduced in article titled "Generalised Non-Negative Matrix Factorisation for Air Pollution Source Apportionment" published in the Science of Total Environment Journal.

Please refer to the article in SoftwareX for more details. 

## Some Features
Uses NumPy (or CuPy, if the package is installed) to factorise a matrix $X_{n\times m}$ into $G_{n\times p}$ and $F_{p\times m}$ with two user defined parameters.

### Covariance Matrix
The covariance matrix ($\Sigma$) should be of size $nm\times nm$ which captures element-wise covariance information. Depending on the type of covariance, GNMF method can act as different methods in the literature,
1. **Covariance Matrix as Identity**: In this case, GNMF method with multiplicative update is same as Lee and Seung's NMF method with multiplicative update. Projected gradient updates can also be used.
2. **Covariance Matrix is Diagonal Matrix**: In this case, GNMF method with multiplicative update is same as LS-NMF method discussed by Wang. Projected gradient updates can also be used.
3. **Covariance Matrix with elements correlated along rows or columns**: In this case, GNMF method with multiplicative update is same as glsNMF method discussed by Plis. Projected gradient updates can also be used.
4. **Covariance Matrix is Dense Matrix**: In this case, the GNMF Method with multiplicative and projected gradient method is as discussed in Lekinwala and Bhushan (2022).

### Number of Factors
A critical parameter for the GNMF to work is the number of factors in which $X$ matrix is to be factorised.

## Functions in `pyGNMF`
Following are some class as part of the module.
1. `gnmf_multiplicative_update`: There are four functions as part of the class,
    - `update_F` : This function is used for the update of F.
    - `update_G` : This function is used for the update of G.
    - `objective_function` : This function is used to compute the value of objective function
    - `running_method` : This function is used to run the method under consideration. Following are the inputs required.

    Use: `gnmf_multiplicative_updaterunning_method(X_matrix, covariance, option=('row_stacked', 'column_stacked'), G_init='random', F_init='random', num_fact=None, num_init=1, max_iter=500000, tolerance=1e-06, conv_typ=('absolute', 'relative'), conv_num=3,)`
    where,
    * `X_matrix`: Matrix to factorise
2. `gnmf_projected_gradient`:
3. `nmf_multiplicative_update`:

