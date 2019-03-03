# Normalizers 
[BaseNormalizer](./base_normalizer.py) defines the interface for all Normalizers.
If new normalizers are added, they should inherit from this class.
For quanser we found that normalizing did not help to solve the environments.  
This is why we only included the [MeanStdNormalizer](./mean_std_normalizer.py) in the final project.