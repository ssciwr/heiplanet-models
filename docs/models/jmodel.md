# Jmodel

This model predicts the basic reproduction number `R0` of the West-Nile virus for a set of temperatures on a given map with geographical coordinates. 

It does so by using externally precomputed `R0` values for different temperatures and interpolates them linearly to the temperatures associated with a given grid point. 

The map can be an ERA5 data file for example, and is expected to be stored in the netcdf format. Temporal and spatial resolution are not restricted. Different restrictions to different regions or resolutions can be passed to the model either by passing it different input maps or by passing different restricting parameters to the `setup_modeldata` function. 

Its use is shown in [the example notebook](../sources/model_jmodel_python.ipynb) in more detail. 
