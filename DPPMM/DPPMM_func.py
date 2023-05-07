#Uses code from https://github.com/ChengzijunAixiaoli/PPMM to build Dynamic PPMM
from mpi4py import MPI
# MPI Fluff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

import sys
import time
import ot
import numpy as np
from scipy.linalg import sqrtm
from DPPMM.PPMM_func import projOtm, Inv
import matplotlib.pyplot as plt
import pickle
from matplotlib import colors
from functools import reduce
import random
from scipy.interpolate import splprep, splev
from KDEpy import FFTKDE



def DPPMM(samples,T,T_interp,N_gen,tol,Nbins,bandwidth):
    tol1,tol2 = tol,tol
   
    p = [len(samples[i]) for i in range(len(samples))]
    N = len(samples[0])
    tnum = len(p)
    _,pp = np.shape(samples[0])
    METHOD = 'SAVE' #method for choosing projection directions. Options are 'SAVE', 'DR', and 'RANDOM'.
    a, b = np.ones((N,)) / N, np.ones((N,)) / N
    T_new = T_interp
    ICmean = 0
    start = np.random.normal(ICmean,.1,(N,pp))
    
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
            itr_data,DIR,lookup = projOtm(itr_data, des_data, N,None, None, method = METHOD,nbins = Nbins,bandwidth = bandwidth)
            lookups.append(lookup)
            directions.append(DIR)
            ddd = itr_data - ori_data
            dist = (np.sqrt(np.mean(np.sum(ddd**2, axis = 1))))
            count += 1
            # if count%100 == 0:
            #     print('PPMM Iteration: ',count, '| Rel: ', np.abs(dist-dist0)/np.abs(dist))
            
        return itr_data,lookups,directions,count
    
    
    def projOtmUtility_gen(data_source,DIR,lookup): #generate output of 1D OT map using current source, direction, and lookup. Function from (https://github.com/ChengzijunAixiaoli/PPMM) 
        ori_proj = np.array(data_source@DIR)
        ori_proj_new = lookup(ori_proj) #*(scale[0]-scale[1])+scale[1]
        delta = ori_proj_new - ori_proj
        res = data_source + np.outer(delta, DIR)
        return res
    
    def generate_samples(ori_data,lookups,directions,itrs): #generate output of OT map using source and collection of directions and 1D lookup tables    
        itr_data = ori_data
        for i in range(itrs):
            DIR = directions[i] 
            lookup = lookups[i]
            itr_data = projOtmUtility_gen(itr_data,DIR,lookup)
        return itr_data
    
    def train_model(tol1,tol2): #train Dynamic_PPMM
        LOOKUPS, DIRECTIONS, COUNTS = [], [], []
        
        '''
        Some calculations to distribute timesteps to ranks
        '''
        tpp = int(tnum/nprocs)
        if tpp == 0: # There are not enough timesteps to parallelize
            if rank == 0:
                for i in range(tnum):
                    if i == 0:
                        ori_data = start
                        tol = tol1
                    else:
                        ori_data = samples[i-1]
                        tol = tol2
                    des_data = samples[i]
                    itr_data,lookups,directions,count = model_samples(ori_data,des_data,tol)
                    LOOKUPS.append(lookups),DIRECTIONS.append(directions),COUNTS.append(count)

            comm.bcast(LOOKUPS,root=0)
            comm.bcast(DIRECTIONS,root=0)
            comm.bcast(COUNTS,root=0)

            return LOOKUPS,DIRECTIONS,COUNTS

        else:
            for i in range(tpp*rank,tpp*(rank+1)):
                
                if i == 0:
                    ori_data = start
                    tol = tol1
                
                else:
                    ori_data = samples[i-1]
                    tol = tol2

                des_data = samples[i]
                itr_data,lookups,directions,count = model_samples(ori_data,des_data,tol)
                LOOKUPS.append(lookups),DIRECTIONS.append(directions),COUNTS.append(count)

            C_LOOKUPS = comm.gather(LOOKUPS,root=0)
            C_DIRECTIONS = comm.gather(DIRECTIONS,root=0)
            C_COUNTS = comm.gather(COUNTS,root=0)


            if rank == 0:
                CC_LOOKUPS, CC_DIRECTIONS, CC_COUNTS = []
                for lst in C_LOOKUPS:
                    CC_LOOKUPS.extend(lst)
                for lst in C_DIRECTIONS:
                    CC_DIRECTIONS.extend(lst)
                for lst in C_COUNTS:
                    CC_COUNTS.extend(lst)

            return CC_LOOKUPS,CC_DIRECTIONS,CC_COUNTS
    
    def test_model(LOOKUPS,DIRECTIONS,COUNTS,n_test): #test Dynamic_PPMM 
        ori_data = np.random.normal(ICmean,.1,(n_test,pp))
        X = np.zeros((tnum,n_test,pp))
        for i in range(tnum):
            itrs = COUNTS[i]
            lookups, directions = LOOKUPS[i], DIRECTIONS[i]
            ori_data = generate_samples(ori_data,lookups,directions,itrs)
            X[i,:,:] = ori_data
        return X
    
    from scipy.interpolate import BSpline, make_interp_spline
    
    
    def interpolate2(X,T,T_new,n_test): #Use transport splines algorithm from (https://arxiv.org/pdf/2010.12101.pdf) to interpolate snapshots
        X_new = np.zeros((len(T_new),n_test,pp))
        from scipy.interpolate import CubicSpline
    
        for i in range(n_test):
            if np.sum(np.isnan(X[:,i,:])) == 0:
                y= X[:,i,:]
                b = make_interp_spline(T, y,bc_type = 'natural')
                X_new[:,i,:] = b(T_new)
            else: 
                X_new[:,i,:] = np.nan
           
        return X_new
        
    import time
    t0 = time.time()
    LOOKUPS,DIRECTIONS,COUNTS = train_model(tol1,tol2)
    t1 = time.time()
    total1 = t1-t0
    n_test = N_gen
    X = test_model(LOOKUPS,DIRECTIONS,COUNTS,n_test)
    X_new = interpolate2(X, T, T_new,n_test) #uncomment to interpolate for some new times T_new
    return X,X_new,total1
