from abc import abstractmethod, ABC


class Controller(ABC):

    @abstractmethod
    def choose_action(self, mu, sigma):
        pass

    @abstractmethod
    def update_params(self, *args):
        pass
