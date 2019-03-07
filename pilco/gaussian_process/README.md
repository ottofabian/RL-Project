# Gaussian Processes

For our experiments, we implemented [normal GPs](gaussian_process.py) completely from scratch and optimize hyperparameters (lengthscales, signal noise, noise variance) with `scipy.minimize`.
The normal GP optimizes a penalized version of the log-likelihood in order to avoid unreasonably large hyperparameters.
Each state dimension has its own GP model, which predicts the change of the current state, and is contained in the wrapper [MultivarianceGP](multivariate_gp.py)

In order to be more computationally efficient, [Sparse GP](sparse_multivariate_gp.py) approximations are implemented based on GPy. 
However, GPy does not allow to optimize custom likelihoods directly,
consequently we constrain the hyperparameter optimization for lengthscales 
between \[0,300\] and for noise variance between \[1e-3, 1e-10\].  
