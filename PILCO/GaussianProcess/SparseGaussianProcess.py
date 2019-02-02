from GPy.models import SparseGPRegression
import numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize


class SparseGaussianProcess(SparseGPRegression):
    """

    TODO: Check if this class is really needed or you can use constraints instead

    Gaussian Process model for regression

    This is a thin wrapper around the SparseGPRegression class, which provides the method _optimize_hyperparams() while
    adding a punisher term to the marginal likelihood term

    :param X: input observations
    :param X_variance: input uncertainties, one per input X
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf+white
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param num_inducing: number of inducing points (ignored if Z is passed, see note)
    :type num_inducing: int
    :rtype: model object

    .. Note:: If no Z array is passed, num_inducing (default 10) points are selected from the data. Other wise num_inducing is ignored
    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self, X, Y, kernel=None, Z=None, num_inducing=10, X_variance=None, mean_function=None, normalizer=None,
                 mpi_comm=None, name='sparse_gp'):
        super(SparseGPRegression, self).__init__(X, Y, kernel, Z, num_inducing, X_variance, mean_function, normalizer,
                                                 mpi_comm, name)

    def optimize(self, optimizer=None, start=None, **kwargs):

        params = np.array(self.kern.rbf.length_scales), np.array(self.kern.rbf.variance), np.array(self.kern.white.variance)

        res = minimize(value_and_grad(self._optimize_hyperparams), params, jac=True, method='L-BFGS-B')

    def _optimize_hyperparams(self, params):
        """
        function handle for scipy optimizer
        :param params: vector of [length scales, signal variance, noise variance]
        :return: penalized marginal log likelihood
        """
        likelihood = -self.log_likelihood()

        # penalty computation
        p = 30
        length_scales, sigma_f, sigma_eps = self.unwrap_params(params)
        std = np.std(self.X, axis=0)

        likelihood = likelihood + (((self.kern.lengthscale - np.log(std)) / np.log(self.length_scale_pen)) ** p).sum()
        likelihood = likelihood + (((sigma_f - sigma_eps) / np.log(self.signal_to_noise)) ** p).sum()
        # print(likelihood)
        return likelihood

    def unwrap_params(self, params) -> tuple:
        """
        unwrap vector of hyperparameters into separate values for gp
        Required for optimization
        :param params: vector of [length scales, signal variance, noise variance]
        :return: length scales, sigma_f, sigma_eps
        """

        return self.kern.rbf.length_scales, self.kern.rbf.variance, self.kern.white.variance
