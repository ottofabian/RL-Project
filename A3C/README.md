# A3C - Asynchronous Advantage Actor-Critic 

Source:\
[Asynchronous Methods for Deep Reinforcement Learning
](https://arxiv.org/abs/1602.01783)


## Best Parameter Settings: 

### Pendulum-v0:  

- network critic: \[3, 100, 1\]
- network actor: \[3, 200, (1,torch.Tensor(\[1e-1\]))\]
- network init: nn.init.normal_(m.weight.data, 0, .1)
- lr: 
    - Actor: .0001
    - Critic: .001
- t_max/batch size: 10
- max episode steps: 200
- gamma: .9
- tau: 1
- beta: 0.01

### CartpoleStabShort:  

Mean reward over 10 episodes: 9999.1377045691

- network critic: \[5, 100, 1\]
- network actor: \[5, 200, (1,1)\]
- network init: nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
- optimizer: 
- max-grad-norm: 
- lr: 
    - Actor: .0001
    - Critic: .001
- t_max/batch size: 50
- max episode steps: 5000
- gamma: .99
- tau: 1
- beta: .01
- max_action: 5
    
### CartpoleStabLong:  
TODO

- lr: 
    - Actor: .0001
    - Critic: .001
- t_max/batch size: 128
- max episode steps: 5000
- gamma: .9
- tau: 1
- beta: .01

### CartpoleSwingShort:  

- lr: 
    - Actor: .00001
    - Critic: .0001
- t_max/batch size: 128
- max episode steps: 5000
- gamma: .999
- tau: 1
- beta: .1

- actor network: 
    - 2 hidden layer 200 nodes

### CartpoleSwingLong:  
TODO

- lr: 
    - Actor: .0001
    - Critic: .001
- t_max/batch size: 128
- max episode steps: 5000
- gamma: .9
- tau: 1
- beta: .01

### Qube/Furuta Pendulum:  
TODO

- lr: 
    - Actor: .0001
    - Critic: .001
- t_max/batch size: 128
- max episode steps: 5000
- gamma: .9
- tau: 1
- beta: .01