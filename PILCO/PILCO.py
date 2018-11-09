import gym
import numpy as np
import quanser_robots

from PILCO.MGPR import MGPR
from PILCO.RBFController import RBFController


class PILCO(object):

    def __init__(self, env_name, seed, n_features):
        # general
        self.env_name = env_name
        self.seed = seed
        self.n_features = n_features
        self.noise_var = None

        # env
        #self.env = quanser_robots.GentlyTerminating(gym.make(self.env_name))

        # check if the requested environment is a quanser robot env
        if self.env_name in ['CartpoleStabShort-v0']:
            self.env = quanser_robots.GentlyTerminating(gym.make(self.env_name))
        else:
            # use the official gym env as default
            self.env = gym.make(self.env_name)

        # get the number of available action from the environment
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.env.seed(self.seed)

        # dynamics model
        # TODO learn length scale by evidence maximization
        self.mgp = MGPR(dim=self.env.observation_space.shape[0])

        self.states = []
        self.actions = []

    def run(self, n_init):
        # sample dataset with random actions
        X = []
        y = []
        rewards = []

        self.noise_var = np.random.normal(0, np.ones(
            self.env.observation_space.shape[0]))  # TODO learn noise variance by evidence maximization
        # self.noise_var = np.diag(np.std(X, axis=1))  # TODO Figure this out

        i = 0
        while i < n_init:
            state_prev = self.env.reset()
            done = False

            while not done or i < n_init:
                action = self.env.action_space.sample()
                state, reward, done, _ = self.env.step(action)

                # state-action pair as input
                X.append(np.append(state_prev, action))

                # delta with following state as output plus some noise
                epsilon = np.random.normal(0, np.abs(self.noise_var))
                y.append(state - state_prev + epsilon)

                rewards.append(reward)

                state_prev = state
                i += 1

        # convert to numpy
        X = np.array(X)
        y = np.array(y)
        rewards = np.array(rewards)

        # init model params

        # dimension of state vector

        print('self.env.observation_space.shape[0]:' + str(self.env.observation_space.shape[0]))
        D = self.env.observation_space.shape[0]
        # dimension of action vector
        F = self.env.action_space

        W = np.random.normal(0, np.identity(self.n_features), size=(D, self.n_features))
        sigma = np.random.normal(0, np.identity(X.shape[1]))
        mu = np.random.normal(0, 1, self.n_features)

        # create controller/policy with those params
        policy = RBFController(W, sigma, mu)

        while True:
            convergence = False
            self.learn_dynamics_model(X, y)
            # TODO model based policy search

            while True:
                self.rollout()
                self.analytic_approximate_policy_evaluation()
                self.policy_improvement()
                W, sigma, mu = self.update_params(W, sigma, mu)

                if convergence:
                    break

            policy.update_params(W, sigma, mu)
            X_test, y_test, reward_test = self.execute_test_run(policy)
            np.append(X, X_test)
            np.append(y, y_test)
            np.append(rewards, reward_test)

            # for debugging
            break

    def learn_dynamics_model(self, X, y):
        self.mgp.fit(X, y)

    def analytic_approximate_policy_evaluation(self):
        # TODO
        # Compute mean and covar of policy/control dist
        # Compute cross covar[x-1, u-1]
        # approx state control dist p(\tildex-1) = p(x-1, u-1) = N(x\tilde-1|mu-1,sigma-1)
        #

        raise NotImplementedError

    def rollout(self, start, policy, episode_len, plant, cost):
        """
        # from: pilco-matlab - https://github.com/ICL-SML/pilco-matlab

        Run multiple rollouts on a trajectory until you reach a terminal state.

        % 1. Generate trajectory rollout given the current policy

        if isfield(plant,'constraint'), HH = maxH; else HH = H; end

        # 1. Generate a trajectory rollout by applying the current policy to the system
        #  The initial state is sampled from p(x0 ) = N (mu0, S0)
        [xx, yy, realCost{j+J}, latent{j}] = ...
          rollout(gaussian(mu0, S0), policy, HH, plant, cost);


        disp(xx);                           % display states of observed trajectory

        x = [x; xx]; y = [y; yy];                            % augment training set

        if plotting.verbosity > 0
          if ~ishandle(3); figure(3); else set(0,'CurrentFigure',3); end
          hold on; plot(1:length(realCost{J+j}),realCost{J+j},'r'); drawnow;
        end


        function [x y L latent] = rollout(start, policy, H, plant, cost)
        %% Code
        % augi indicies for variables augmented to the ode variables

        if isfield(plant,'augment'), augi = plant.augi;             % sort out indices!
        else plant.augment = inline('[]'); augi = []; end

        if isfield(plant,'subplant'), subi = plant.subi;
        else plant.subplant = inline('[]',1); subi = []; end

        odei = plant.odei; poli = plant.poli; dyno = plant.dyno; angi = plant.angi;
        simi = sort([odei subi]);
        nX = length(simi)+length(augi); nU = length(policy.maxU); nA = length(angi);

        state(simi) = start; state(augi) = plant.augment(state);      % initializations
        x = zeros(H+1, nX+2*nA);
        x(1,simi) = start' + randn(size(simi))*chol(plant.noise);
        x(1,augi) = plant.augment(x(1,:));
        u = zeros(H, nU); latent = zeros(H+1, size(state,2)+nU);
        y = zeros(H, nX); L = zeros(1, H); next = zeros(1,length(simi));

        for i = 1:H % --------------------------------------------- generate trajectory
          s = x(i,dyno)'; sa = gTrig(s, zeros(length(s)), angi); s = [s; sa];
          x(i,end-2*nA+1:end) = s(end-2*nA+1:end);

          % 1. Apply policy ... or random actions --------------------------------------
          if isfield(policy, 'fcn')
            u(i,:) = policy.fcn(policy,s(poli),zeros(length(poli)));
          else
            u(i,:) = policy.maxU.*(2*rand(1,nU)-1);
          end
          latent(i,:) = [state u(i,:)];                                  % latent state

        :return:
        """

        # check for system constraint boundaries
        #self.env.action_space.high
        #self.env.action_space.low

        # For our gym-environments we don't need the ODE because we already get sufficient env-feedback
        #idces_ode_solver = None
        #idces_agugmenting_ode = None
        #idces_dynamics_out = None
        # ...

        # variable
        x = np.zeros([episode_len + 1, self.state_dim + 2 * self.n_actions])
        # x[0, odei] = multivariate_normal(start, plant.noise)

        actions_u = np.zeros((episode_len, self.n_actions))
        targets_y = np.zeros((episode_len, self.state_dim))
        loss_l = np.zeros(episode_len)
        latent = np.zeros((episode_len + 1, self.n_features + self.state_dim))

        # run multiple rollouts along a trajectory for episode_len number of steps.
        for i in range(episode_len):
            raise NotImplementedError

        return x, targets_y, loss_l, latent

    def policy_improvement(self):
        raise NotImplementedError

    def update_params(self, W, sigma, mu):
        # use CG or L-BFGS for updates
        # TODO
        return W, sigma, mu

    def execute_test_run(self, policy):
        X = []
        y = []
        rewards = []

        state_prev = self.env.reset()
        done = False

        while not done:
            # TODO
            action = policy.predict(state_prev)
            state, reward, done, _ = self.env.step(action)

            # create history and create new training instance
            X.append(np.append(state_prev, action))
            epsilon = np.random.normal(0, self.noise_var)
            y.append(state - state_prev + epsilon)

            state_prev = state

        return X, y, reward

    # def compute_deltas(self, x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    #     """
    #     Returns the deltas between the states plus some noise
    #     :param x: sequence of recorded states
    #     :param sigma: diagonal matrix containing the std for each state index
    #     """
    #     deltas = []
    #
    #     for i, x1 in enumerate(x):
    #         epsilon = np.random.normal(0, sigma)
    #         if i == 0:
    #             deltas.append(x1 + epsilon)
    #         deltas.append(x1 - x[i - 1] + epsilon)
    #
    #     return deltas
