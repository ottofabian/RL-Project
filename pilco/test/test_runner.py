from pilco.test.test_controller import test_rbf, test_squash, test_linear, test_set_params_linear, test_set_params_rbf
from pilco.test.test_cost import test_cost, test_trajectory_cost
from pilco.test.test_grad import test_grad_mgpr, test_grad_smgpr, test_grad_rollout, test_grad_loss, test_grad_squash
from pilco.test.test_prediction import test_mgpr, test_smgpr
from pilco.test.test_rollout import test_rollout

if __name__ == '__main__':
    test_mgpr()
    test_smgpr()
    test_squash()
    test_rbf()
    test_linear()
    test_set_params_linear()
    test_set_params_rbf()
    test_rollout()
    test_cost()
    test_trajectory_cost()
    test_grad_mgpr()
    test_grad_smgpr()
    test_grad_rollout()
    test_grad_loss()
    test_grad_squash()
