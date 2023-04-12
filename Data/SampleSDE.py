# Simulate sample paths of an SDE using the Euler-Maruyama method
# N is the dimension, F is the forcing parameter, Ntimes is the number of snapshots, T is the final time, and Npts is the number of points per snapshot
import numpy as np


def Sample_SDE(syst,Npts):
    
    if syst == 'vanderpol':
        N = 2  # Number of variables, i.e., dimension
        T = 6
        Ntimes = 101
        diffusion = 0.0025
        noise_start = .05
        start = 1
        dt = T/(Ntimes-1)
        

    if syst == 'OU':
        N = 30
        Ntimes = 101
        beta = .1
        diffusion = .05
        noise_start = .5
        start = 10
        T = 15
        dt = T/(Ntimes-1)


    if syst == 'L96':
        N = 30 #Number of variables, i.e., dimension
        F = 2  # Forcing
        T = 3.5
        Ntimes = 101
        diffusion = 0.0025
        noise_start = .05
        start = 4
        dt = T/(Ntimes-1)


    def L96(x):
        Y = np.zeros((Npts,N))
        for i in range(N):
            Y[:,i] = (x[:,(i + 1) % N] - x[:,i - 2]) * x[:,i - 1] - x[:,i] + F
        return Y

    def OU(x):
        Y = np.zeros((Npts,N))
        for i in range(N):
            Y[:,i] = -beta*x[:,i]
        return Y

    def vanderpol(x):
        mu = 1
        Y = np.zeros((Npts,N))
        Y[:,0] = x[:,1]
        Y[:,1] = mu*(1-x[:,0]**2)*x[:,1] - x[:,0]
        return Y

    def sample_path(system):
        X = np.zeros((Npts,N))
        R = np.random.normal(0,1,(Npts,N))
        J = np.random.choice([-1,1],(Npts))
        if syst == 'vanderpol':
            system = vanderpol
        if syst == 'L96':
            system = L96
        if syst == 'OU':
            system = OU
        if system == OU:
            X[:,0] = start*J+noise_start*R[:,0]
            X[:,1:] = start + noise_start*R[:,1:]
        if system == L96:
            X[:,0] = start + noise_start*R[:,0]
            X[:,1:] =  noise_start*R[:,1:]
        if system == vanderpol:
            X = start + noise_start*R  
        W = np.random.normal(0, 1, (Ntimes,Npts, N))
        Xs = np.zeros((Ntimes,Npts, N))
        t = 0
        ts = []
        for i in range(Ntimes):
            ts.append(t)
            Xs[i,:, :] = X
            X = X + system(X)*dt+W[i,:, :]*np.sqrt(2*diffusion*dt)
            t += dt
        return Xs, ts
    Xs, T = sample_path(syst)
    Xs = Xs[::10]
    Ts= T[::10]
    m = np.max(np.abs(Xs))
    Xs = Xs/m
    Xs = np.array(Xs)
    return Xs, Ts