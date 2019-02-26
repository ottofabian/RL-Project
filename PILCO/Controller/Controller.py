from abc import abstractmethod, ABC


class Controller(ABC):

    @abstractmethod
    def choose_action(self, mu, sigma, bound):
        pass

    @abstractmethod
    def set_params(self, params):
        pass
