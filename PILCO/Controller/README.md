# Controller

We only use the RBF controller for our experiments as it is universally applicable for all problems. 
Linear controllers are too limited for most applications, but can be sufficient for stabilization around an equilibrium. [see](https://ieeexplore.ieee.org/document/6654139)
Consequently, the CartpoleStabShort-v0 task could be solved with this,
however in our experiments the RBF policy was also able to solve the environment after only 2 episodes and the inital random samples.  
New controllers can be implemented by inheriting from the [Controler](./Controller.py) class.

## RBF Controller
We define our RBF controller as deterministic Gaussian Process based on the original implementation from [Deisenroth](https://ieeexplore.ieee.org/document/6654139).
The signal variance is fixed to 1 and the noise variance to 0.01, which results in a signal-to-noise ration of 10.
The lengthscales are initialized with 1. 
The majority of the parameters are pseudo samples with trainable inputs and targets for the GP fit.
Further, we squash the action output of the RBF Policy with the sin transformation to an predefined symmetric action range.
