# Simulate sample paths of an SDE using the Euler-Maruyama method
# N is the dimension, F is the forcing parameter, Ntimes is the number of snapshots, T is the final time, and Npts is the number of points per snapshot

import matplotlib.pyplot as plt
import numpy as np
import pickle

syst = 'vanderpol' #L96, vanderpol, lorenz
def Sample_SDE(syst):
    
    if syst == 'lorenz':
        N = 3  # Number of variables, i.e., dimension
        T = 1
        Ntimes = 100
        Npts = 100
        diffusion = 1
        noise_start = 0
        start = 25
        dt = T/Ntimes
        
    if syst == 'vanderpol':
        N = 2  # Number of variables, i.e., dimension
        T = 10
        Ntimes = 100
        Npts = 100
        diffusion = 0.01
        noise_start = .1
        start = 1
        dt = T/Ntimes
        
    if syst == 'L96':
        N = 10  # Number of variables, i.e., dimension
        F = -2 # Forcing
        T =5
        Ntimes =   1000
        Npts = 100
        diffusion = .001
        noise_start = 0.0
        start = -2
        dt = T/Ntimes
        
    def L96(x):
        d = np.zeros(N)
        for i in range(N):
            d[i] = (x[(i + 1) %N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d
    
    def vanderpol(x):
        mu = 1
        return np.array([x[1],mu*(1-x[0]**2)*x[1]-x[0]]).T
    
    
    def lorenz(x):
        s,r,b = 10,28,8/3
        return np.array([s*(x[1]-x[0]),x[0]*(r-x[2])-x[1],x[0]*x[1]-b*x[2]])
    
    def sample_path(system): 
        if syst == 'vanderpol':
            system = vanderpol
        if syst == 'lorenz':
            system = lorenz
        if syst == 'L96':
            system = L96
        if system == L96:           
            x = start+noise_start*np.random.normal(0,1,(N))
        if system == lorenz:
            x = [0,0,start]+noise_start*np.random.normal(0,1,(N))
        if system == vanderpol:
            x = start+noise_start*np.random.normal(0,1,(N))
        W = np.random.normal(0,1,(Ntimes,N))
        X = np.zeros((Ntimes,N))
        t = 0
        ts = []
        for i in range(Ntimes):
            ts.append(t)
            x = x + system(x)*dt+W[i,:]*np.sqrt(2*diffusion*dt)
            X[i,:] = x
            t += dt
        return X, ts
        
    Xs = []
    for i in range(Npts):
        X,T = sample_path(syst)
        Xs.append(X)     
    m = np.max(np.abs(Xs))
    Xs = Xs/m
    Xs = np.array(Xs)
    return Xs,m,T,dt,diffusion
   

Xs,m,T,dt,diffusion = Sample_SDE(syst)
T = np.array(T)
plt.show()
plt.scatter(Xs[:,:,0],Xs[:,:,1],s = .1)

with open("samples.p", "wb") as f:
    pickle.dump([Xs*m,T,dt,diffusion], f)
    

    
        
    