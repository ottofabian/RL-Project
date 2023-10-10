# Reinforcement Learning Project

Technische Universität Darmstadt, Winter Semester 2018/2019  <img align="right" src="resources/general/TU_logo.png" width="128">

Supervision: [Jan Peters](https://www.ias.informatik.tu-darmstadt.de/Member/JanPeters), [Samuele Tosatto](https://www.ias.informatik.tu-darmstadt.de/Team/SamueleTosatto)


## Authors
- [Johannes Czech](https://github.com/QueensGambit)
- [Fabian Otto](https://github.com/ottofabian)

## Algorithms
- [A3C](a3c/README.md)
- [PILCO](pilco/README.md)

## Platforms
- Cartpole Stabilization ([Further info](https://www.google.com/search?source=hp&ei=EQffW4yLJYPKwQKQjoOIAQ&q=Cart-pole+stabilization&btnK=Google+Search&oq=Cart-pole+stabilization&gs_l=psy-ab.3...480.480..991...0.0..0.85.85.1......0....1j2..gws-wiz.ns_kSRav_wc))
- Cartpole Swing-up ([Further info](https://www.google.com/search?source=hp&ei=EQffW4yLJYPKwQKQjoOIAQ&q=Cart-pole+swing-up&btnK=Google+Search&oq=Cart-pole+swing-up&gs_l=psy-ab.3..0i22i30.730.730..901...0.0..0.123.123.0j1......0....1j2..gws-wiz.sjBBp2UuE9A))
- Qube/Furuta Pendulum ([Further info](https://www.google.com/search?source=hp&ei=EQffW4yLJYPKwQKQjoOIAQ&q=Furuta+pendulum+swing-up&btnK=Google+Search&oq=Furuta+pendulum+swing-up&gs_l=psy-ab.3..0i22i30.716.716..808...0.0..0.64.64.1......0....1j2..gws-wiz.roZTOV-jxVs))

## Installation

The following Python packages are required:
- autograd
- baselines - For installation details see: https://github.com/openai/baselines
- dill 
- GPy
- gym
- matplotlib
- matplotlib2tikz (Optional in case PILCO plots should be saved)
- numpy
- pytorch
- scipy
- tensorboard
- tensorboardX
- tensorflow
- torchvision
- [quanser_robots](https://git.ias.informatik.tu-darmstadt.de/quanser/clients)

The following Linux packages are required:
- ffmpeg

The following is required for PILCO Test cases:
- Octave installation
- oct2py (python package)

We also offer to install all required packages directly through our [anaconda environment export](./conda_env.yml).

For creating a new [anaconda](https://anaconda.org/anaconda/python) environment based on a YML-file use:
```bash
conda env create --name my_env_name --file path/to/conda_env.yml python=3.6.5
```

## Experiments

__Please be aware that the Quanser environments are still subject to change and results or policies might not be reproducible or applicable anymore.__
__The latest Quanser version introduced different constraints for the cartpole environment which can cause issues.__ 

We added a small subset of experiment runs, which we found useful in order to get a better feeling for hyper-parameters and the algorithm in general. 
This allows to compare different hyper-parameter settings, performance and sample efficiency. 

More details can be found [here](./experiments/README.md).  
In order to run experiments with [A3C](a3c/README.md) or [PILCO](pilco/README.md), please check the corresponding README.

Log files for all runs will be saved to `./experiments/logs/`.

## Citation
Our comprehensive report can be found [here](./Czech_Otto_Lab_Report.pdf)
```bibtex
@software{otto_czech_2019,  
	title = {Project Lab Reinforcement Learning, {TU} Darmstadt, {WS}18/19: {ottofabian}/{RL}-Project},  
	url = {https://github.com/ottofabian/RL-Project},  
	shorttitle = {Project Lab Reinforcement Learning, {TU} Darmstadt, {WS}18/19},  
	author = {{Otto, Fabian and Czech, Johannes}},  
	urldate = {2019-03-15},  
	date = {2019-03-15},  
}
```

