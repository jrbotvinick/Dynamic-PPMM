import argparse
import numpy as np
import torch
from matplotlib import cm, pyplot as plt
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from ConditionedRQNSF.resnet import ResidualNet, ConvResidualNet
from ConditionedRQNSF.mlp import MLP
from ConditionedRQNSF.torchutils import create_alternating_binary_mask,get_num_parameters,tensor2numpy
import pickle
import torch.nn as nn
from nflows import transforms, distributions
from nflows.distributions.normal import ConditionalDiagonalNormal
import random
from ConditionedRQNSF.base import Flow
from ConditionedRQNSF.data_loader import build_batches
import nflows.utils as utils
import math


def RQ_NSF(trainingsamples,T,T_interp,N_gen,train_time,architecture):
        
        
    # _,T_interp,_,_ = data2
    
    # m = np.max(np.abs(trainingsamples))
    # trainingsamples = trainingsamples/m

    T = np.array(T)
    samples = trainingsamples
    p = [len(samples[i]) for i in range(len(samples))]

    T = np.array(T)
    N = max([len(i) for i in trainingsamples])
    # M = max([len(i) for i in trainingsamples])

    tnum = len(p)
    _,pp = np.shape(samples[0])
    parser = argparse.ArgumentParser()
    dim = pp
    
    if architecture == 'normal':
        Layers = 10
    if architecture == 'fast':
        Layers = 2

        
    training_steps = 1000000
    # model
    parser.add_argument('--num_bins', type=int, default=8)
    parser.add_argument('--learning_rate', default=5e-4)
    parser.add_argument('--num_training_steps', default=int(training_steps))
    parser.add_argument('--grad_norm_clip_value', type=float, default=5.)
    parser.add_argument('--monitor_interval', default=1)
    parser.add_argument('--seed', default=1264)
    args = parser.parse_args()
    # torch.manual_seed(args.seed)
    # random.seed(2154)
    device = torch.device('cpu')
    # create model
    
    def create_linear_transform():
            return transforms.CompositeTransform([
                transforms.RandomPermutation(features=pp),
                transforms.LULinear(pp, identity_init=True)
            ])
    def create_base_transform(i):
            return transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=create_alternating_binary_mask(features=dim, even=(i % 2 == 0)),
                transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=128,
                    context_features = 1,
                    num_blocks=2,
                    use_batch_norm=True
                ),
                tails='linear',
                tail_bound=3,
                num_bins=args.num_bins,
                apply_unconditional_transform=False
            )

    

    if architecture == 'normal':
        transform = transforms.CompositeTransform([
            transforms.CompositeTransform([
                create_linear_transform(),
                create_base_transform(i)
            ]) for i in range(Layers)
        ] + [
            create_linear_transform()
        ])
    if architecture == 'reduced':
        transform = transforms.CompositeTransform([
            transforms.CompositeTransform([
                create_linear_transform(),
                create_base_transform(i)
            ]) for i in range(2)
        ] + [
            create_linear_transform()
        ])   

        
    ####################### Make embedding net
    embedding_net = MLP(
        in_shape=[1],
        out_shape=[1],
        hidden_sizes=[100,100,100,100],
        activation=nn.ReLU(),
        activate_output=False)
    
    embedding_net2 = MLP(
        in_shape=[1],
        out_shape=[2*dim],
        hidden_sizes=[500], 
        activation=nn.ReLU(),
        activate_output=False)
    # ##########################
    
    distribution = ConditionalDiagonalNormal(shape=[dim], 
                                        context_encoder=embedding_net2)

    # distribution = distributions.StandardNormal(shape=[dim])
    # distribution = Normal2(shape = [dim])
    flow = Flow(transform, distribution,embedding_net).to(device)
    n_params = get_num_parameters(flow)
    print('There are {} trainable parameters in this model.'.format(n_params))
    # create optimizer
    epochs = 5
    # thresh = 200
    # batch_size = 10
    batch_size = 10 #for fish, 10 else
    optimizer = optim.Adam(flow.parameters(), lr=args.learning_rate)
    # optimizer2 = optim.Adam(embedding_net2.parameters(), lr=args.learning_rate)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs*N//batch_size, 0)
    # scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, thresh, 0)

    totn = 0
    for j in range(len(trainingsamples)):
        for k in range(len(trainingsamples[j])):
            totn+=1
            
    def train():  
        tbar = tqdm(range(args.num_training_steps))
        all_loss = []
        bcount = 0
        bbtime = 0
        ecount = 0
        t0 = time.time()
        for step in tbar:
            etime =  time.time() - bbtime - t0
            # print(etime,train_time)
            if etime >= train_time:
                break
            
            if step % (N//batch_size) == 0:
                BATCH_SAMPLES,T_BATCHES,btime = build_batches(trainingsamples,T,batch_size)   
                bcount = 0
                bbtime += btime
            if ecount % (N//batch_size*epochs) == 0:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs*N//batch_size, 0)
                ecount = 0
            batch_samples = torch.tensor(BATCH_SAMPLES[bcount],dtype = torch.float).reshape(batch_size*tnum,pp)
            cond_time = torch.tensor(T_BATCHES[bcount],dtype = torch.float)
            flow.train()    
            # embedding_net2.requires_grad_(False)
            optimizer.zero_grad()
            # if bcount > thresh:
            #     embedding_net2.requires_grad_(False)
            #     scheduler2.step()
            #     optimizer2.zero_grad()
                
            # m,s= distribution._compute_params(torch.tensor(T,dtype = torch.float).reshape(len(T),1))
            # print(m[0])

            log_density = flow.log_prob(batch_samples,context = cond_time)
            loss = - torch.mean(log_density)
            all_loss.append(loss.detach().numpy())
            loss.backward()
            bcount+=1
            ecount += 1
            if args.grad_norm_clip_value is not None:
                clip_grad_norm_(flow.parameters(), args.grad_norm_clip_value)
            optimizer.step()
            scheduler.step()

            if (step) % args.monitor_interval == 0:
                s = 'Loss: {:.4f}'.format(loss.item())
                tbar.set_description(s)  
            
                # print(scheduler.get_last_lr())
                # print(scheduler2.get_last_lr())

            
        return all_loss,etime


    def model(T_new):
        # torch.manual_seed(123451263)
        print('Testing')
        flow.eval()
        Xs = np.zeros((len(T_new),N_gen,pp))
        with torch.no_grad():
            for i in range(len(T_new)):
                samples = flow.sample(N_gen, context = torch.tensor(T_new[i],dtype = torch.float).reshape(1,1))
                samples = tensor2numpy(samples).reshape(N_gen,pp)
                Xs[i,:,:] = samples
        return np.array(Xs)
                    
    
    import time
    all_loss,ttime = train()

    X1 = model(T)
    X2 = model(T_interp)

    
    return X1,X2,ttime