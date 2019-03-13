from abc import abstractmethod, ABC
import autograd.numpy as np
import pickle


class Controller(ABC):

    @abstractmethod
    def choose_action(self, mean: np.ndarray, cov: np.ndarray, bound: np.ndarray = None) -> tuple:
        pass

    @abstractmethod
    def set_params(self, params: np.ndarray):
        pass

    @abstractmethod
    def get_params(self):
        pass

    def save_policy(self, save_dir) -> None:
        """
        pickle dumps the policy to hard drive
        :param save_dir: directory where the policy will be saved
        :return: None
        """
        pickle.dump(self, open(f"{save_dir}policy.p", "wb"))
