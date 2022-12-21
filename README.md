# Dynamic-PPMM

This repository adapts the PPMM optimal transport solver [1] to the time-dependent setting by joining samples from successive snapshots by optimal transport maps and applying the transport splines algorithm [2] to interpolate between snapshots using cubic splines. We also include the implementation of a rational quadratic neural spline flow conditioned on time, which can be used as a baseline comparison for Dynamic PPMM. Here is a summary of the files in the repository:
- `Sample_SDE.py`: Sample inference snapshot data using the Euler-Maruyama method. 
- `PPMM_func.py`: Base functions used in the original PPMM implementation [1].
- `Dynamic_PPMM.py`: Uses `PPMM_func.py` to learn optimal transport plans between `Sample_SDE`'s snapshot data and implements the transport splines algorithm [2]. 
- `torchutils.py`: PyTorch utility functions from [3].
- `resnet.py`: Residual network from [3].
- `mlp.py`: Multi-Layer perceptron from [3].
- `base.py`: Flow operations from [3].
- `data_loader.py`: Load batched data for training of time-conditioned neural spline flow.
- `Conditioned_RQ_NSF`: Conditions a neural spline flow to learn the evolving density from `Sample_SDE`'s snapshot data, using [3] and [4]. 


[1] PPMM: https://github.com/ChengzijunAixiaoli/PPMM \
[2] Transport splines: https://arxiv.org/abs/2010.12101 \
[3] Neural spline flows: https://github.com/bayesiains/nsf \
[4] nflows: https://github.com/bayesiains/nflows



