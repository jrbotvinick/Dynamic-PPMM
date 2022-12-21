#Partitions dataset into batches for RQ-NSF training
import argparse
import numpy as np
import torch
from matplotlib import cm, pyplot as plt
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from resnet import ResidualNet, ConvResidualNet
from torchutils import create_alternating_binary_mask,get_num_parameters,tensor2numpy
import pickle
import torch.nn as nn
from nflows import transforms, distributions
from base import Flow
import random
import statistics 

with open("samples.p", "rb") as f:
    data = pickle.load(f)
    
   
trainingsamples, T, dt = data[0],data[1], data[2]
N,tnum,pp = np.shape(trainingsamples)
T = np.array(T)

def load_data_random(trainingsamples,T,numpts): #this function can be used to randomly choose a subset of the data for each step of the optimization
    N,tnum,pp = np.shape(trainingsamples)
    r = np.random.choice(range(tnum),numpts,replace = False)
    s = random.sample(range(N),numpts)
    return trainingsamples[s,r,:], T[r]
    
def build_batches(trainingsamples,T,Nper): #this function splits trainingsamples into batches for the RQ-NSF training. Each batch includes Nper samples from every observation time. 
    N,tnum,pp = np.shape(trainingsamples)
    n_batches = N//Nper
    shuffled = np.zeros((N,tnum,pp))
    for j in range(len(T)):
        shuffled[:,j,:] = random.sample(list(trainingsamples[:,j,:]),N)
    Xbatches = []
    for j in range(n_batches):
        Xbatches.append(shuffled[j*Nper:(j+1)*(Nper),:,:])
    return Xbatches
