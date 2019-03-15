# Models

For our experiments we evaluated a [shared network architecture](actor_critic_network.py),
which has a shared body and three 3 outputs representing mean and standard deviation of the action normal distribution as well the value of the state.
Further, we found that splitting [actor](actor_network.py) and [critic](critic_network.py) performed better.  
Adding a [LSTM layer](actor_critic_lstm.py) did not help for the quanser environments due to the state's full observability.
This is why we do not support the LSTM network in our implementation. 
