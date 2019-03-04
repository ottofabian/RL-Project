from abc import abstractmethod, ABC


class Loss(ABC):

    @property
    @abstractmethod
    def target_state(self):
        pass

    @property
    @abstractmethod
    def state_dim(self):
        pass

    @abstractmethod
    def compute_loss(self, mu, sigma):
        raise NotImplementedError

    @abstractmethod
    def compute_cost(self, mu, sigma):
        raise NotImplementedError
