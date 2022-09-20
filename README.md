# Multivariate Postprocessing of Temporal Dependencies with Autoregressive and LSTM Neural Networks

by
Daniel Tolomei,
Sjoerd Dirksen, 
Kirien Whan, 
Maurice Schmeits

## Abstract
Weather forecasts issued by Numerical Weather Prediction (NWP) systems often display systematic bias and do not quantify the inherent uncertainty of the forecast. It is the task of statistical postprocessing to use these NWP predictions to issue probabilistic forecasts that address these issues. In this work we focus on multivariate postprocessing, which also requires statistical modelling of the spatial, temporal or inter-variable dependencies. More especifically, we use NWP forecasts from the Harmonie-Arome model to issue multivariate probabilistic forecasts for hourly wind speed predictions from initialization at 0 UTC up to 48h ahead. We propose a new statistical model for multivariate forecasting, the ARMOS(p) model, that exploits the autoregressive property of forecast errors to estimate an explicit parametric distribution, and compare it to a benchmark obtained from a combination of Ensemble Output Statistics (EMOS) with the Schaake Shuffle. We further extend these models by estimating the distribution parameters using neural networks, which incorporate spatial and temporal information from the NWP forecasts by using LSTM and Convolutional layers. In our experiments we verify model performance by computing proper multivariate scores and by performing marginal verification on the test set. The results show that the LSTM/EMOSnet and the ARMOS(2)net improve on the benchmarks, and are the best models overall.


## Dependencies
Developed in Python3, all libraries inclued in the requirements.txt file.
To run code, create a virtual environment and install libraries by running the following.
```
python3 -m venv pyenv
source pyenv/bin/activate   
pip install -r requirements.txt
```

It is also necessary to export the PYTHONPATH environment variable so that src is treated as a Python package.
```
export PYTHONPATH=$PWD
```


## References
https://studenttheses.uu.nl/handle/20.500.12932/41500?show=full