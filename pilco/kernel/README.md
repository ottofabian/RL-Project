# Kernels

We use [RBF kernels](rbf_kernel.py) combined with a [white noise kernels](white_noise_kernel.py) for the GP computation. 
These classes are only used for the normal GP, the sparse GP uses the implementation from GPy. 
Adding new Kernels is possible by inherting from the [Kernel](kernel.py) class.