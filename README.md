# Dynamic-PPMM

This repository adapts the PPMM optimal transport solver to the time-dependent setting by joining samples from successive snapshots by optimal transport maps and applying the transport splines algorithm to interpolate between snapshots with cubic splines. We also include the implementation of a rational quadratic neural spline flow conditioned on time, which can be used as a baseline comparison for Dynamic PPMM. Here is a summary of the files in the repository:
- `Sample_SDE.py`:
- `PPMM_func.py`:
- `Dynamic_PPMM.py`:
- `torchutils.py`:
- `resnet.py`:
- `mlp.py`:
- `base.py`:
- `data_loader.py`:
- `Conditioned_RQ_NSF`:
