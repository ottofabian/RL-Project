# Utils
We define several utils functions to make the main code more readable.
Here you can find the following functionalities: 
- [sync_grads](./util.py#sync_grads) for synchronizing the local worker gradients with the global network in order to update the global network   
- [save_checkpoint](./util.py#save_checkpoint) for saving a model checkpoint
- [load_saved_optimizer](./util.py#load_saved_optimizer) loads previously stored optimizer from given path
- [load_saved_model](./util.py#load_saved_model) loads previously stored model from given path
- [get_model](./util.py#get_model) get model instance, required for handling different types of models as specified [here](../Models/README.md)
- [get_optimizer](./util.py#get_optimizer) return optimizer instance without shared statistics, supports split and shared model
- [get_shared_optimizer](./util.py#get_shared_optimizer) return optimizer instance without shared statistics, supports split and shared model
- [get_normalizer](./util.py#get_normalizer) get normalizer instance
- [make_env](./util.py#make_env) get callable to create env instance
- [log_to_tensorboard](./util.py#log_to_tensorboard) log training info to tensorboard 
- [parse_args](./util.py#parse_args) parse console arguments 