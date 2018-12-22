# A3C - Asynchronous Advantage Actor-Critic 

Source:\
[Asynchronous Methods for Deep Reinforcement Learning
](https://arxiv.org/abs/1602.01783)


## Best Parameter Settings: 

### CartpoleStabShort:  

- lr: 
    - Actor: .00001
    - Critic: .0001
- t_max/batch size: 128
- max episode steps: 5000
- gamma: .99
- tau: 1
- beta: .01
    
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
- t_max/batch size: 256
- max episode steps: 5000
- gamma: .995
- tau: 1
- beta: .05

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