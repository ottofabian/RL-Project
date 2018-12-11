from PILCO.Test.Test_Controller import test_rbf, test_squash
from PILCO.Test.Test_Cost import test_cost, test_trajectory_cost
from PILCO.Test.Test_Prediction import test_predictions
from PILCO.Test.Test_Rollout import test_rollout

if __name__ == '__main__':
    test_predictions()
    test_squash()
    test_rbf()
    test_rollout()
    test_cost()
    test_trajectory_cost()
