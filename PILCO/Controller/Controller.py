from abc import abstractmethod, ABC


class Controller(ABC):

    @abstractmethod
    def choose_action(self, mu, sigma):
        pass

    # @abstractmethod
    # def set_hyper_params(self, *args):
    #     pass
    #
    # @abstractmethod
    # def get_hyperparams(self):
    #     pass
