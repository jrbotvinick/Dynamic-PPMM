#Adapted PPMM_func from (https://github.com/ChengzijunAixiaoli/PPMM)

import numpy as np
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
from scipy import ndimage



#==============================================================================
############# Fast covariance matrix calculation ##############
### x: 2-d dataframe, n*p
#Reference for weighted covariance matrix
#https://link.springer.com/referenceworkentry/10.1007%2F978-3-642-04898-2_612
def fastCov(data, weight):
    data_weight = data * weight.reshape(-1, 1)
    data_mean = np.mean(data_weight, axis = 0)
    sdata = (data - data_mean)*np.sqrt(weight).reshape(-1, 1)
    data_cov = sdata.T.dot(sdata)/(data.shape[0]-1)
    return data_cov    
#==============================================================================

import random


#==============================================================================
############### SAVE direction #################
### x, y: 2-d array
def saveDir(x_ori, y_ori, ws, wt): 
    nn,_ = np.shape(x_ori)
    mm,_ = np.shape(y_ori)
    pp = x_ori.shape[1]
    data_bind = np.concatenate((x_ori, y_ori))
    weight_bind = np.concatenate((ws, wt))
    data_cov = fastCov(data_bind, weight_bind)
    covinv = np.linalg.inv(data_cov)
    signrt = sqrtm(covinv)
    
    
    data_weight = data_bind * weight_bind.reshape(-1, 1)
    cm = data_weight.mean(axis = 0)
    #cm = data_bind.mean(axis = 0)
    v1 = fastCov((x_ori-cm)@signrt, ws)
    v2 = fastCov((y_ori-cm)@signrt, wt)
    
    diag = np.diag(np.repeat(1, pp))
    savemat = ((v1-diag)@(v1-diag) + (v2-diag)@(v2-diag))/2
    eigenValues, eigenVectors = np.linalg.eig(savemat)
    idx = eigenValues.argsort()
    vector = eigenVectors[:, idx[-1]]
    dir_temp = signrt@vector
    return dir_temp/np.sqrt(dir_temp@dir_temp)

#==============================================================================
############### Directional regression (DR) #################
### x, y: 2-d array
def drDir(x_ori, y_ori, ws, wt):
    
    pp = x_ori.shape[1]
    data_bind = np.concatenate((x_ori, y_ori))
    weight_bind = np.concatenate((ws, wt))
    data_cov = fastCov(data_bind, weight_bind)
    covinv = np.linalg.inv(data_cov)
    signrt = sqrtm(covinv)
    
    data_weight = data_bind * weight_bind.reshape(-1, 1)
    cm = data_weight.mean(axis = 0)
    #cm = data_bind.mean(axis = 0)
    s1 = (x_ori-cm)@signrt*ws.reshape(-1, 1)
    s2 = (y_ori-cm)@signrt*wt.reshape(-1, 1)
    e1 = s1.mean(axis = 0)
    e2 = s2.mean(axis = 0)
    v1 = fastCov((x_ori-cm)@signrt, ws)
    v2 = fastCov((y_ori-cm)@signrt, wt)
    
    mat1 = ((v1 + np.outer(e1, e1))@(v1 + np.outer(e1, e1)) 
            + (v2 + np.outer(e2, e2))@(v2 + np.outer(e2, e2)))/2
    mat2 = (np.outer(e1, e1) + np.outer(e2, e2))/2
    
    diag = np.diag(np.repeat(1, pp))
    drmat = 2*mat1 + 2*mat2@mat2 + 2*sum(np.diag(mat2))*mat2 - 2*diag
    eigenValues, eigenVectors = np.linalg.eig(drmat)
    idx = eigenValues.argsort()[::-1] 
    vector = eigenVectors[:, idx[0]]
    dir_temp = signrt@vector
    #dir_temp = signrt@np.linalg.eig(drmat)[1][:,0]
    return dir_temp/np.sqrt(dir_temp@dir_temp)
#==============================================================================
    



#==============================================================================
############# uniform to sphere ##############
### vec: 1-d array
def uniform2sphere(vec):
    p = len(vec)
    vec_temp = 1.-2.*vec
    vec_temp[0] = 2.*np.pi*vec[0]
    x_temp = np.array([np.cos(vec_temp[0]), np.sin(vec_temp[0])])
    if p==1:
        return x_temp
    else:
        for i in range(1,p):
            xx_temp = np.append(np.sqrt(1-vec_temp[i]**2)*x_temp, vec_temp[i])
            x_temp = xx_temp
        return x_temp    
#==============================================================================
    
    
    
    

def Inv(x):
# =============================================================================
    rank_x = np.argsort(np.argsort(x))
    res = np.array(range(len(x)))[rank_x]
# =============================================================================
    # ww = weight[np.argsort(x)]
    # rank_x = np.argsort(np.argsort(x)) #This works as the 'order' function in R
    # res = ((np.cumsum(ww) - ww/2)/sum(ww)*len(x))[rank_x]
    return res
import matplotlib.pyplot as plt
import statsmodels.api as sm # recommended import according to the docs
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


from scipy.stats import norm, gaussian_kde
from scipy.special import ndtr
from KDEpy import FFTKDE

from statsmodels.stats.weightstats import DescrStatsW
import dill


def projOtmUtility(data_source, data_target,N, ws, wt,DIR,nbins,bandwidth):
    x = np.array(data_source@DIR)
    y = np.array(data_target@DIR) 
    r2= np.max(np.array([np.max(x),np.max(y)]))
    r1 = np.min(np.array([np.min(x),np.min(y)]))
    L = .1
    pts = np.linspace(r1-L,r2+L,nbins)
    kernel = 'gaussian'
    estimator1 = FFTKDE(kernel=kernel, bw=bandwidth)
    estimator2 = FFTKDE(kernel=kernel, bw=bandwidth)
    pdf1 = estimator1.fit(x).evaluate(pts)
    pdf2 = estimator2.fit(y).evaluate(pts)
    eps = 1e-8
    pdf1 += eps
    pdf2 += eps  
    pdf1,pdf2 = pdf1/np.sum(pdf1),pdf2/np.sum(pdf2)
    cdf1 = np.cumsum(pdf1)
    cdf2 = np.cumsum(pdf2)
    F = interp1d(pts,cdf1,kind = 'linear',bounds_error = False, fill_value = None)
    G = interp1d(cdf2,pts,kind = 'linear',bounds_error = False, fill_value = None)
    def T(x): #transport map T
        y = G(F(x))
        y[np.isnan(y)] = x[np.isnan(y)]
        return y
    ori_proj_new = T(x)
    delta = ori_proj_new - x
    res = data_source + np.outer(delta, DIR)

    T_serialized = dill.dumps(T)

    return res,T_serialized

#Projected one-dimensional optimal transport        
def projOtm(data_source, data_target, N,weight_source= None, weight_target= None, method= "SAVE", nbins = 10,bandwidth='scott'):   
    if weight_source is None:
        weight_source = np.repeat(1, data_source.shape[0])
    if weight_target is None:
        weight_target = np.repeat(1, data_target.shape[0])
    if method == "SAVE":
        DIR = saveDir(data_source, data_target, weight_source, weight_target)
    if method == "DR":
        DIR = drDir(data_source, data_target, weight_source, weight_target)
    if method == "RANDOM":
        vec = np.random.uniform(size = np.shape(data_source)[1]-1)
        DIR = uniform2sphere(vec)
    res,lookup = projOtmUtility(data_source, data_target, N,weight_source, weight_target, DIR,nbins,bandwidth) 
    return res,DIR,lookup



        

#==============================================================================