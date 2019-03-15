# Utils
We define several utils functions to make the main code more readable.
Here you can find the following functionalities: 
- [sync_grads](./util.py#L28) for synchronizing the local worker gradients with the global network in order to update the global network   
- [save_checkpoint](./util.py#42) for saving a model checkpoint
- [load_saved_optimizer](./util.py#61) loads previously stored optimizer from given path
- [load_saved_model](./util.py#82) loads previously stored model from given path
- [get_model](./util.py#L106) gets model instance, required for handling different types of models as specified [here](../models/README.md)
- [get_optimizer](./util.py#L140) returns optimizer instance without shared statistics, supports split and shared model
- [get_shared_optimizer](./util.py#L168) return optimizer instance without shared statistics, supports split and shared model
- [get_normalizer](./util.py#L215) get normalizer instance
- [make_env](./util.py#L242) gets callable to create env instance
- [log_to_tensorboard](./util.py#265) logs training info to tensorboard 
- [shape_rewards](./util.py#331) reshapes rewards if required
- [parse_args](./util.py#352) parses console arguments 
