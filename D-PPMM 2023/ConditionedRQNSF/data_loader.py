import numpy as np
import torch
import random
import time

def build_batches(trainingsamples,T,Nper): #this function splits trainingsamples into batches for the RQ-NSF training. Each batch includes Nper samples from every observation time. Requires that Nper divides N = max(len(trainingsamples[j])) 
    s = time.time()
    N = max([len(i) for i in trainingsamples]) 
    _,pp = np.shape(trainingsamples[0])
    Xbatches = []
    Tbatches = []
    ids = []
    Tt = []
    for j in range(len(T)):
        for i in range(Nper):
            Tt.append(T[j])
    for j in range(len(trainingsamples)):
        ll = len(trainingsamples[j])
        li = []
        num = N//ll
        for k in range(num):
            li.append(random.sample(range(0,ll),ll))
        if N/ll > num:
            li.append(random.sample(range(0,ll),N-ll*num))
        flat_list = []
        for sublist in li:
            for item in sublist:
                flat_list.append(item)
        ids.append(flat_list)
    for k in range(N//Nper):
        Xbatches.append(np.array([trainingsamples[j][ids[j][Nper*k:Nper*(k+1)]] for j in range(len(trainingsamples))]).reshape(Nper,len(trainingsamples),pp))
        Tbatches.append(torch.tensor(Tt, dtype = torch.float).reshape(len(T)*Nper,1))
    f = time.time()
    return Xbatches,Tbatches,f-s
