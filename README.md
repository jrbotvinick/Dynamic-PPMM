# Dynamic-PPMM

This repository contains Python code which extends the PPMM optimal transport solver [1] to the time-dependent setting by joining samples from successive snapshots by optimal transport maps and applying the transport splines algorithm [2] to interpolate between snapshots using cubic splines. We also include the implementation of a rational quadratic neural spline flow conditioned on time, which can be used as a baseline comparison for Dynamic PPMM. Here is a summary of the relevant files:

- `ConditionedRQNSF`: Conditions a neural spline flow on time to learn the evolving density from `Sample_SDE`'s snapshot data, using code from [3] and [4]. 
- `DPPMM`: Implementation of Dynamic-PPMM. Uses code from [1] and [2] to learn optimal maps between the inference snapshot data and interpolate the evolving density with transport splines.
- `Data`: Generates inference SDE sample path snapshot data. 
- `main.py`: Example file which shows how to generate the SDE snapshot data and run both D-PPMM and time-conditioned RQ-NSF to interpolate the measurements. 

The code in this repository is compatible with 

- `Python` 3.9
- `torch` 1.11.0
- `numpy` 1.20.3
- `matplotlib` 3.5.1

  
The following video shows an example of D-PPMM with transport spline interpolation applied to the 2-dimensional van der Pol oscillator. 


https://user-images.githubusercontent.com/100333155/231488452-337a8a8d-0f64-4dce-8490-ac20e37ef80d.mp4


The following video shows an example of D-PPMM with transport spline interpolation applied to a 10-dimensional OU-process. 



https://user-images.githubusercontent.com/100333155/231482397-49278b7f-f366-420d-9ff4-6b9ac96f9250.mp4



The following video shows an example of D-PPMM with transport spline interpolation applied to the 10-dimensional Lorenz-96 system. 



https://user-images.githubusercontent.com/100333155/231486946-607ca9ae-420e-4bb9-a673-f6c00aa6f308.mp4



The following video shows an example of D-PPMM with transport spline interpolation applied to the fish schooling dataset [5]. 

https://user-images.githubusercontent.com/100333155/231466983-081da8b5-b4cc-4d60-8423-20bb5d80edd0.mp4





[1] PPMM: https://github.com/ChengzijunAixiaoli/PPMM \
[2] Transport splines: https://arxiv.org/abs/2010.12101 \
[3] Neural spline flows: https://github.com/bayesiains/nsf \
[4] nflows: https://github.com/bayesiains/nflows \
[5] Fish schooling dataset: https://ir.library.oregonstate.edu/concern/datasets/zk51vq07c






