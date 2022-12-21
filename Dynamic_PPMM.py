#Uses code from https://github.com/ChengzijunAixiaoli/PPMM to build Dynamic PPMM
import sys
import time
import ot
import numpy as np
from scipy.linalg import sqrtm
from PPMM_func import projOtm, Inv
import matplotlib.pyplot as plt
import pickle
from matplotlib import colors
from functools import reduce
import random
from scipy.interpolate import splprep, splev

with open("samples.p", "rb") as f: #load snapshot data from SDE_loader
    data = pickle.load(f)

samples, T, dt, _ = data[0],data[1], data[2], data[3]
tol = 1e-4 #\alpha

m = np.max(np.abs(samples))
samples = samples/m #normalize data
METHOD = 'SAVE' #method for choosing projection directions. Options are 'SAVE', 'DR', and 'RANDOM'.
N,tnum,pp = np.shape(samples)
a, b = np.ones((N,)) / N, np.ones((N,)) / N


def model_samples(ori,des,tol): #train PPMM at one snapshot pair
    lookups = []
    directions = []
    dist0 = 0
    dist = 1
    count = 0
    ori_data = ori
    itr_data = ori_data
    while np.abs(dist-dist0)/np.abs(dist) > tol:
        des_data = des
        dist0 = dist
        itr_data,DIR,lookup = projOtm(itr_data, des_data, a, b, method = METHOD)
        lookups.append(lookup)
        directions.append(DIR)
        ddd = itr_data - ori_data
        dist = (np.sqrt(np.mean(np.sum(ddd**2, axis = 1))))
        count += 1
        if count%100 == 0:
            print('PPMM Iteration: ',count, '| Rel: ', np.abs(dist-dist0)/np.abs(dist))
        
    return itr_data,lookups,directions,count


def projOtmUtility_gen(data_source, ws, wt, DIR,lookup): #generate output of 1D OT map using current source, direction, and lookup. Function from (https://github.com/ChengzijunAixiaoli/PPMM) 
    ori_proj = np.array(data_source@DIR)   
    itr_temp = Inv(ori_proj, ws)
    ori_proj_new = lookup(itr_temp)
    delta = ori_proj_new - ori_proj
    res = data_source + np.outer(delta, DIR)
    return res

def generate_samples(ori_data,lookups,directions,itrs): #generate output of OT map using source and collection of directions and 1D lookup tables    
    itr_data = ori_data
    for i in range(itrs):
        DIR = directions[i]
        lookup = lookups[i]
        itr_data = projOtmUtility_gen(itr_data, a, b,DIR,lookup)
    return itr_data

def train_model(tol): #train Dynamic_PPMM
    LOOKUPS, DIRECTIONS, COUNTS = [], [], []
    for i in range(tnum):
        if i == 0:
            ori_data = np.random.normal(0,1,(N,pp))
        else:
            ori_data = samples[:,i-1,:]
        des_data = samples[:,i,:]
        itr_data,lookups,directions,count = model_samples(ori_data,des_data,tol)
        LOOKUPS.append(lookups),DIRECTIONS.append(directions),COUNTS.append(count)
        print('Training timestep: ', i)
    return LOOKUPS,DIRECTIONS,COUNTS

def test_model(LOOKUPS,DIRECTIONS,COUNTS): #test Dynamic_PPMM
    ori_data = np.random.normal(0,1,(N,pp))
    X = np.zeros((N,tnum,pp))
    for i in range(tnum):
        itrs = COUNTS[i]
        lookups, directions = LOOKUPS[i], DIRECTIONS[i]
        ori_data = generate_samples(ori_data,lookups,directions,itrs)
        X[:,i,:] = ori_data
        print('Testing timestep: ', i)
    return X

def interpolate(X,T,T_new): #Use transport splines algorithm from (https://arxiv.org/pdf/2010.12101.pdf) to interpolate snapshots
    X_new = np.zeros((N,len(T_new),pp))
    for i in range(N):
        y = X[i,:,:].T
        tck, _ = splprep(y,u=T, s=0)
        X_new[i,:,:] = np.array(splev(T_new, tck)).T
    return X_new

    
import time
t0 = time.time()
LOOKUPS,DIRECTIONS,COUNTS = train_model(tol)
t1 = time.time()
total1 = t1-t0
t0 = time.time()

X = test_model(LOOKUPS,DIRECTIONS,COUNTS)
# X_new = interpolate(X, T, T_new) #uncomment to interpolate for some new times T_new
t1 = time.time()
total2 = t1-t0
plt.scatter(X[:,:,0],X[:,:,1],s = .1)
plt.xlim(-.75,.75)
plt.ylim(-1,1)
plt.show()
with open("Dynamic_PPMM_samples.p", "wb") as f:
    pickle.dump([T,X,m,total1], f)

print('Time to Train: ',total1)
print('Time to Sample: ',total2)
