#Uses code from nflows (https://github.com/bayesiains/nflows) and nsf (https://github.com/bayesiains/nsf) to construct time-conditioned RQ-NSF
import argparse
import numpy as np
import torch
from matplotlib import cm, pyplot as plt
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from resnet import ResidualNet, ConvResidualNet
from mlp import MLP
from torchutils import create_alternating_binary_mask,get_num_parameters,tensor2numpy
import pickle
import torch.nn as nn
from nflows import transforms, distributions
from base import Flow
from data_loader import build_batches
import nflows.utils as utils

with open("samples.p", "rb") as f:
    data = pickle.load(f) #load inference data (need to run SDE_loader)
    
train_time = .45 #set max training time (s)
architecture = 'fast' #choose 'normal' or 'fast' 

trainingsamples, T, dt = data[0],data[1], data[2]
m = np.max(np.abs(trainingsamples))
trainingsamples = trainingsamples/m

T = np.array(T)
N,tnum,pp = np.shape(trainingsamples)
bounds = [[-1,1],[-1.25,1.25]]
parser = argparse.ArgumentParser()
dim = pp
batch_size = 2 #number of observations at each timestep in each batch. Total batchsize is therefore len(T)*batch_size.

if architecture == 'normal':
    Layers = 10
if architecture == 'fast':
    Layers = 2
    
training_steps = 100 #max number of trainingsteps for flow

# model parameters
parser.add_argument('--num_bins', type=int, default=8)
parser.add_argument('--learning_rate', default=5e-4)
parser.add_argument('--num_training_steps', default=int(training_steps))
parser.add_argument('--grad_norm_clip_value', type=float, default=5.)
parser.add_argument('--monitor_interval', default=1)
parser.add_argument('--seed', default=1638128)
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cpu')

# create model
distribution = distributions.StandardNormal((dim,))
              
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
if architecture == 'fast':
    transform = transforms.CompositeTransform([
        create_base_transform(i) for i in range(2)
    ])
####################### Make embedding net to condition on time 
embedding_net = MLP(
    in_shape=[1],
    out_shape=[1],
    hidden_sizes=[100,100],
    activation=nn.ReLU(),
    activate_output=False)
##########################
flow = Flow(transform, distribution,embedding_net).to(device)
n_params = get_num_parameters(flow)
print('There are {} trainable parameters in this model.'.format(n_params))

# create optimizer
optimizer = optim.Adam(flow.parameters(), lr=args.learning_rate)
def train():  #train the conditional RQ-NSF 
    tbar = tqdm(range(args.num_training_steps))
    all_loss = []
    bcount = 0 
    t0 = time.time()
    cond_time = torch.tensor(np.tile(T,batch_size), dtype = torch.float).reshape(len(T)*batch_size,1)
    for step in tbar:
        if step % (N//batch_size) == 0:
            BATCH_SAMPLES = build_batches(trainingsamples,T,batch_size)   
            bcount = 0
        batch_samples = torch.tensor(BATCH_SAMPLES[bcount].reshape(batch_size*tnum,pp),dtype = torch.float)
        bcount+=1
        flow.train()
        optimizer.zero_grad()
        log_density = flow.log_prob(batch_samples,context = cond_time)
        loss = - torch.mean(log_density)
        all_loss.append(loss.detach().numpy())
        if time.time() - t0 > train_time:
            break
        loss.backward()
        if args.grad_norm_clip_value is not None:
            clip_grad_norm_(flow.parameters(), args.grad_norm_clip_value)
        optimizer.step()
        if (step) % args.monitor_interval == 0:
            s = 'Loss: {:.4f}'.format(loss.item())
            tbar.set_description(s)      
    return all_loss


def model(T_new):
    print('Testing')
    flow.eval()
    Xs = np.zeros((N,len(T),pp))
    with torch.no_grad():
        for i in range(len(T)):
            samples = flow.sample(N, context = torch.tensor(T_new[i],dtype = torch.float).reshape(1,1))
            samples = tensor2numpy(samples).reshape(N,pp)
            Xs[:,i,:] = samples
    return np.array(Xs)
                

import time
t0 = time.time()    
all_loss = train()
t1 = time.time()
total1 = t1 - t0
t0 = time.time()
X = model(T)
#X = model(T_interp) #uncomment to interpolate between T at times T_interp (which need to be defined)
t1 = time.time()
total2 = t1-t0

plt.scatter(X[:,:,0],X[:,:,1],s = .1)
plt.xlim(-.75,.75)
plt.ylim(-1,1)
plt.show()

with open("NF_samples.p", "wb") as f:
    pickle.dump([T,X], f)
    

print('Time to Train: ',total1)
print('Time to Sample: ',total2)
