# pyGNMF
 Python Library for Generalised Non-Negative Matrix Factorisiation (GNMF)


This Python library implements GNMF method introduced in article titled "Generalised Non-Negative Matrix Factorisation for Air Pollution Source Apportionment" published in the Science of Total Environment Journal.

Following are the details of the code,

$\min_{G,F} J = & \dfrac{1}{2}\left[(X_v - (GF)_v)^\intercal \Sigma_v^{-1}(X_v - (GF)_v)\right]
			\text{s.t. }   & G_{i, l}, F_{l, j} \geq 0\\
			               & \forall ~i=1,2,\cdots,n;\nonumber
			~j=1,2,\cdots,m;\nonumber
			~l=1,2,\cdots,p \nonumber
$
