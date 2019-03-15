# Utils
We define several utils functions to make the main code more readable.
Here you can find the following functionalities: 
- [sync_grads](./util.py#L25) for synchronizing the local worker gradients with the global network in order to update the global network   
- [save_checkpoint](./util.py#39) for saving a model checkpoint
- [load_saved_optimizer](./util.py#58) loads previously stored optimizer from given path
- [load_saved_model](./util.py#L79) loads previously stored model from given path
- [get_model](./util.py#L103) gets model instance, required for handling different types of models as specified [here](../models/README.md)
- [get_optimizer](./util.py#L137) returns optimizer instance without shared statistics, supports split and shared model
- [get_shared_optimizer](./util.py#L165) return optimizer instance without shared statistics, supports split and shared model
- [get_normalizer](./util.py#L212) get normalizer instance
- [make_env](./util.py#L232) gets callable to create env instance
- [log_to_tensorboard](./util.py#L255) logs training info to tensorboard 
- [shape_rewards](./util.py#L221) reshapes rewards if required
- [parse_args](./util.py#L348) parses console arguments 
