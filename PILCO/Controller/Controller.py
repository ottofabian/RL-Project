from abc import abstractmethod, ABC


class Controller(ABC):

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def update_params(self, *args):
        pass
