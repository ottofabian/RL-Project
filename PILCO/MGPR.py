import numpy as np
import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class MGPR(GaussianProcessRegressor):
    """
    Multivariate Gaussian Process Regression
    """

    def __init__(self, dim, optimizer="fmin_l_bfgs_b", length_scales=1., sigma_f=1, sigma_eps=1, alpha=1e-10,
                 n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None):

        """

        :param dim: number of GPs to use, should be equaivalent to the number of
        :param optimizer:
        :param length_scale:
        :param sigma_f:
        :param sigma_eps:
        :param alpha:
        :param n_restarts_optimizer:
        :param normalize_y:
        :param copy_X_train:
        :param random_state:
        """

        # TODO: Check if WhiteKernel is used correctly
        ridge = 1e-6
        kernel = RBF(length_scale=np.ones(len(length_scales))) + WhiteKernel(ridge)

        super(MGPR, self).__init__(kernel, alpha, optimizer, n_restarts_optimizer, normalize_y, copy_X_train,
                                   random_state)
        self.dim = dim
        self.length_scales = length_scales
        # For a D-dimensional state space, we use D separate GPs, one for each state dimension.
        # - Efficient Reinforcement Learning using Gaussian Processes, Marc Peter Deisenroth
        self.gp_container = [
            GaussianProcessRegressor(kernel, alpha, optimizer, n_restarts_optimizer, normalize_y, copy_X_train,
                                     random_state) for _ in range(dim)]

    def fit(self, X, Y):
        for i in range(self.dim):
            self.gp_container[i].fit(X, Y[:, i])

    def predict(self, X, return_std=False, return_cov=False):
        Y = np.empty((X.shape[0], self.dim))
        stat = []

        for i in range(self.dim):
            if return_cov or return_std:
                Y[:, i], tmp = self.gp_container[i].predict(X, return_std=return_std, return_cov=return_cov)
                stat.append(tmp)
            else:
                res = self.gp_container[i].predict(X, return_std=return_std, return_cov=return_cov)
                Y[:, i] = res
        if stat:
            return Y, np.array(stat)

        return Y

    def get_kernels(self, X):
        """
        Returns the gram matrix built out of all gaussian process containers.
        """
        gram_matrix = np.zeros((len(self.gp_container), len(X), len(X)))

        # concatenate all individual gram matrices for each gaussian process
        for i in range(len(self.gp_container)):
            gram_matrix[i] = self.gp_container[i].kernel_(X)  # .diag(X)

        return gram_matrix

    def y_train_mean(self):
        means = []
        for i in range(self.dim):
            means.append(self.gp_container[i].y_train_mean())
        return means

    def sample_y(self, X, n_samples=1, random_state=0):
        samples = []
        for i in range(self.dim):
            samples.append(self.gp_container[i].sample_y(X, n_samples, random_state))
        return samples

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        lml = []
        for i in range(self.dim):
            lml.append(self.gp_container[i].log_marginal_likelihood(theta, eval_gradient))
        return lml

    def get_alphas(self):
        return np.array([c.alpha_ for c in self.gp_container])

    def predict_on_noisy_inputs(self, m, s):
        iK, beta = self.calculate_factorizations()
        return self.predict_given_factorizations(m, s, iK, beta)

    # TODO: make this work for noisy input, aviods writing the gp completely from scratch
    # https: // github.com / nrontsis / PILCO / blob / e7307d0c4e6687f09892643ac63fa29576bba7cf / pilco / models / mgpr.py

    def calculate_factorizations(self):
        K = self.kernel(X)
        batched_eye = np.eye(np.shape(self.X)[0], batch_shape=[self.dim], dtype=float_type)
        L = np.linalg.cholesky(K + self.noise[:, None, None] * batched_eye)
        iK = scipy.linalg.cho_solve(L, batched_eye)
        y_ = y.T[:, :, None]
        # Why do we transpose Y? Maybe we need to change the definition of self.Y() or beta?
        beta = np.linalg.cho_solve(L, y_)[:, :, 0]
        return iK, beta

    def predict_given_factorizations(self, m, s, iK, beta):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action
             and inv(s)*input-ouputcovariance
        """

        s = np.tile(s[None, None, :, :], [self.dim, self.dim, 1, 1])
        inp = np.tile(self.centralized_input(m)[None, :, :], [self.dim, 1, 1])

        # Calculate M and V: mean and inv(s) times input-output covariance
        iL = np.diag(1 / self.)
        iN = inp @ iL
        B = iL @ s[0, ...] @ iL + np.eye(self.num_dims, dtype=float_type)

        # Redefine iN as in^T and t --> t^T
        # B is symmetric so its the same
        t = (np.linalg.solve(B, np.linalg.transpose(iN), adjoint=True)).T

        lb = np.exp(-np.reduce_sum(iN * t, -1) / 2) * beta
        tiL = t @ iL
        c = self.variance / np.sqrt(np.linalg.det(B))

        M = (np.reduce_sum(lb, -1) * c)[:, None]
        V = np.matmul(tiL, lb[:, :, None], adjoint_a=True)[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        R = s @ np.matrix_diag(
            1 / np.square(self.length_scales[None, :, :]) +
            1 / np.square(self.length_scales[:, None, :])
        ) + np.eye(self.num_dims, dtype=float_type)

        # TODO: change this block according to the PR of tensorflow. Maybe move it into a function?
        X = inp[None, :, :, :] / np.square(self.length_scales[:, None, None, :])
        X2 = -inp[:, None, :, :] / np.square(self.length_scales[None, :, None, :])
        Q = np.linalg.solve(R, s) / 2
        Xs = np.reduce_sum(X @ Q * X, -1)
        X2s = np.reduce_sum(X2 @ Q * X2, -1)
        maha = -2 * np.matmul(X @ Q, X2, adjoint_b=True) + \
               Xs[:, :, :, None] + X2s[:, :, None, :]
        #
        k = np.log(self.variance)[:, None] - \
            np.reduce_sum(np.square(iN), -1) / 2
        L = np.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        S = (np.tile(beta[:, None, None, :], [1, self.dim, 1, 1])
             @ L @
             np.tile(beta[None, :, :, None], [self.dim, 1, 1, 1])
             )[:, :, 0, 0]

        diagL = np.transpose(np.linalg.diag_part(np.transpose(L)))
        S = S - np.diag(np.reduce_sum(np.multiply(iK, diagL), [1, 2]))
        S = S / np.sqrt(np.linalg.det(R))
        S = S + np.diag(self.variance)
        S = S - M @ np.transpose(M)

        return np.transpose(M), S, np.transpose(V)

    def centralized_input(self, m):
        return self.X - m
