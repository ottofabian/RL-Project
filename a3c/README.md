# A3C - Asynchronous Advantage Actor-Critic 

This is our implementation of A3C and the corresponding synchronous version A2C based on the paper [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) from Mnih, et al.
We also combined this with [General Advantage Estimation](https://arxiv.org/abs/1506.02438) as it has shown improved performance for policy gradient methods.   

Our code structure is defined the following:
- [models](models): Neural Network models for actor and critic.  
- [optimizers](optimizers): Optimizers with shared statistics for A3C.  
- [util](./util): Helper methods to make main code more readable.


## Executing experiments
1) Activate the anaconda environment
```bash
source activate my_env
```
2) Execute the [A3CRunner](../a3c_runner.py) script (the default environment is CartpoleStabShort-v0)

Training run from scratch:
```bash
python3 my/path/to/a3c_runner.py
```

Training run from an existing policy:
```bash
python3 my/path/to/a3c_runner.py --path my_model_path
```

More console arguments (e.g. hyperparameter changes) can be added to the run, for details see
```bash
python3 my/path/to/a3c_runner.py --help
```

3) (Optional) Start tensorboard to monitor training progress
```bash
tensorboard --logdir=./Experiments/runs 
```

## Executing evaluation run for existing policy
1) Activate the anaconda environment
```bash
source activate my_env
```

2) Execute the [A3CRunner](../a3c_runner.py) script
```bash
python3 my/path/to/a3c_runner.py --path my_model_path --test
``
