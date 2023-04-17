from DPPMM.DPPMM_func import DPPMM
from Data.SampleSDE import Sample_SDE
from ConditionedRQNSF.RQ_NSF_func import RQ_NSF
import matplotlib.pyplot as plt
import numpy as np

X,T = Sample_SDE('vanderpol',Npts = 10000) #Generate Inference Trajectories
Tinterp = np.linspace(0,max(T)-.01,100) #Interpolation times 


X1,X2,training_time = DPPMM(X,T,Tinterp,500,1e-4,100,'scott') 
X3,X4, training_time2 = RQ_NSF(X,T,Tinterp,500,training_time,'reduced') 


for j in range(len(X)):
    plt.scatter(X[j,:,0],X[j,:,1],s = .1,c = 'k')
plt.title('Inference Data')
plt.show()

for j in range(len(X1)):
    plt.scatter(X1[j,:,0],X1[j,:,1],s = .1,c = 'k')
plt.title('D-PPMM Samples')
plt.show()

for j in range(len(X2)):
    plt.scatter(X2[j,:,0],X2[j,:,1],s = .1,c = 'k')
plt.title('D-PPMM Interpolation')
plt.show()

for j in range(len(X3)):
    plt.scatter(X3[j,:,0],X3[j,:,1],s = .1,c = 'k')
plt.title('NF Samples')
plt.show()

for j in range(len(X4)):
    plt.scatter(X4[j,:,0],X4[j,:,1],s = .1,c = 'k')
plt.title('NF Interpolation')
plt.show()