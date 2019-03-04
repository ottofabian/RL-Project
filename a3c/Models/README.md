# Models

For our experiments we evaluated a [shared network architecture](./ActorCriticNetwork.py),
which has a shared body and three 3 outputs representing mean and standard deviation of the action normal distribution as well the value of the state.
Further, we found that splitting [actor](./ActorNetwork.py) and [critic](./CriticNetwork.py) performed better.
Adding a [LSTM layer](./ActorCriticLSTM.py) did not help for the quanser environments due to the state's full observability.