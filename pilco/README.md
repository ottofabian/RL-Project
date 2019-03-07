# PILCO — Probabilistic Inference for Learning COntrol

This is our implementation of [PILCO](http://mlg.eng.cam.ac.uk/pilco/) from Deisenroth, et al.  
The implementation is largely based on the [matlab code](https://github.com/ICL-SML/pilco-matlab) and the [PhD thesis](https://www.google.de/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiR4sHA6ejgAhVSzaQKHaPRAt4QFjAAegQIChAB&url=https%3A%2F%2Fwww.ksp.kit.edu%2Fdownload%2F1000019799&usg=AOvVaw1zhWQ8A31UbT_oR7E2kP07) of Deisenroth. 

Our code structure is defined the following:
- [Controller](controller): Controller/policy models.
- [CostFunctions](cost_function): Cost functions for computing a trajectories performance.
- [GaussianProcess](gaussian_process): (Sparse) Gaussian Process models for learning dynamics and RBF policy. 
- [Kernels](kernel): Kernel functions for Gaussian Process models.
- [Test](test): Test cases to ensure the implementation is working as intended.   
- [util](./util): Helper methods to make main code more readable.


## Executing experiments
1) Activate the anaconda environment
```bash
source activate my_env
```
2) Execute the [PILCORunner](../pilco_runner.py) script (the default environment is CartpoleStabShort-v0)

Training run from scratch:
```bash
python3 my/path/to/PILCORunner.py
```

Training run from an existing policy:
```bash
python3 my/path/to/PILCORunner.py --weight-dir my_model_directory
```

More console arguments (e.g. hyperparameter changes) can be added to the run, for details see
```bash
python3 my/path/to/PILCORunner.py --help
```

## Executing evaluation run for existing policy
1) Activate the anaconda environment
```bash
source activate my_env
```

2) Execute the [PILCORunner](../pilco_runner.py) script
```bash
python3 my/path/to/PILCORunner.py --weight-dir my_model_directory --test
```
