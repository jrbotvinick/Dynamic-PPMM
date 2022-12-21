# Dynamic-PPMM

This repository adapts the PPMM optimal transport solver to the time-dependent setting by joining samples from successive snapshots by optimal transport maps and applying the transport splines algorithm to interpolate between snapshots with cubic splines. We also include the implementation of a rational quadratic neural spline flow conditioned on time, which can be used as a baseline comparison for Dynamic PPMM. Here is a summary of the files in the repository:
- `Sample_SDE.py`: Sample inference snapshot data using the Euler-Maruyama method. 
- `PPMM_func.py`: Base functions used in the original PPMM implementation [X]
- `Dynamic_PPMM.py`: Uses `PPMM_func.py` to learn optimal transport plans between `Sample_SDE`'s snapshot data. 
- `torchutils.py`: PyTorch utility functions from [X]
- `resnet.py`: Residual network from [X]
- `mlp.py`: Multi-Layer perceptron from [X]
- `base.py`: Flow operations from [X]
- `data_loader.py`: Load batched data for training of time-conditioned neural spline flow.
- `Conditioned_RQ_NSF`: Conditions a neural spline flow to learn the evolving density from `Sample_SDE`'s snapshot data. 
