# Reinforcement Learning Project
Technische Universität Darmstadt winter semester 2018/2019  
Supervision: Jan Peters, Samuele Tosatto

## Authors
- Johannes Czech
- Fabian Otto

## Algorithms
- [A3C](./A3C/README.md)
- [PILCO](./PILCO/README.md)

## Platforms
- Cartpole stabilization ([Further info](https://www.google.com/search?source=hp&ei=EQffW4yLJYPKwQKQjoOIAQ&q=Cart-pole+stabilization&btnK=Google+Search&oq=Cart-pole+stabilization&gs_l=psy-ab.3...480.480..991...0.0..0.85.85.1......0....1j2..gws-wiz.ns_kSRav_wc))
- Cartpole swing-up ([Further info](https://www.google.com/search?source=hp&ei=EQffW4yLJYPKwQKQjoOIAQ&q=Cart-pole+swing-up&btnK=Google+Search&oq=Cart-pole+swing-up&gs_l=psy-ab.3..0i22i30.730.730..901...0.0..0.123.123.0j1......0....1j2..gws-wiz.sjBBp2UuE9A))
- Qube ([Further info](https://www.google.com/search?source=hp&ei=EQffW4yLJYPKwQKQjoOIAQ&q=Furuta+pendulum+swing-up&btnK=Google+Search&oq=Furuta+pendulum+swing-up&gs_l=psy-ab.3..0i22i30.716.716..808...0.0..0.64.64.1......0....1j2..gws-wiz.roZTOV-jxVs))

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
- [quanser robots](https://git.ias.informatik.tu-darmstadt.de)

The following Linux packages are required:
- ffmpeg

The following is required for PILCO Test cases:
- Octave installation
- oct2py (python package)

We also offer to install all required packages directly through our [anaconda environment export](./conda_env.yaml).

In order to run Experiments with [A3C](./A3C/README.md) or [PILCO](./PILCO/README.md), please check the corresponding README.


