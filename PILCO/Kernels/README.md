# Kernels

We use [RBF kernels](./RBFKernel.py) combined with a [white noise kernels](./WhiteNoiseKernel.py) for the GP computation. 
These classes are only used for the normal GP, the sparse GP uses the implementation from GPy. 
Adding new Kernels is possible by inherting from the [Kernel](./Kernel.py) class.