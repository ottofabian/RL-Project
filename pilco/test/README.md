# Test

Idea of test is taken from: https://github.com/nrontsis/PILCO/edit/master/tests/

This directory contains test cases for the majority of the PILCO codebase:  
- [Controller, sin squash and parameter changes](./test_controller.py)
- [Multivariate (sparse) GP predictions](./test_prediction.py)
- [Saturating cost function](./test_cost.py)
- [Trajectory rollout computation](./test_rollout.py)
- [Gradients](./test_grad.py)

Running all test is possible by executing:
```bash
pytest path/to/test_runner.py
``` 

It is likely that some gradient test will fail.
PILCO can easily run into numerical issues throughout simple matrix multiplications
which results in the gradient difference being below the `HIPS/autograd` default precision of 1-e6 
compared to the numerical computation.
