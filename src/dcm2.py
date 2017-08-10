"""
Created on Thu Dec  1 12:51:52 2016

@author: alejandro
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import random as rn
import nibabel as nb

def get_stimulus(N_pts=300, TR=1, nblocks=5, duration_range=(1,3), dist_opt='random'):
    ''' This function generates a pattern of stimulus '''
    if dist_opt == 'random':
        timeline = np.arange(0, TR * N_pts, TR)    
        time_offset = 10
        possible_onsets = np.arange(time_offset, N_pts-time_offset)
        sep_bwn_ons = 10
        durations = []
        onsets = []
        for i in range(nblocks):
            current_block_duration = rn.randint(duration_range[0], duration_range[1])
            current_onset = rn.choice( possible_onsets )
            onsets.append( current_onset )
            durations.append( current_block_duration )
            possible_onsets = possible_onsets[np.logical_or( possible_onsets < current_onset - sep_bwn_ons,
                                                    possible_onsets > current_onset + sep_bwn_ons)]
        u = np.zeros(N_pts)
        for i,d in zip(onsets, durations):
            u[i:i+d] = 1
        return u, timeline, onsets, durations
    elif dist_opt == 'fixed':
        return 'not implemented'

def hrf(t, tau1=5.4, delta1=6.0, tau2=10.8, delta2=12.0, c=0.35):  
    t = np.copy(t)
    gamma1 = ((t/tau1)**delta1) * np.exp(-(delta1/tau1)*(t-tau1))
    gamma2 = c*((t/tau2)**delta2) * np.exp(-(delta2/tau2)*(t-tau2))
    h = gamma1 - gamma2
    return h

def Hmat(h):
    fr = np.zeros( len(h) )
    fr[0] = h[0]
    H = toeplitz(h, fr)
    return H   
   
def get_dcm_data(SNR=4, N_nds=5, N_pts=300, TR=1, A=None, N_cluster = 1000):
    #SNRs = [0.2,1.0,4.0]
    #SNR = 4
    #N_nds = 5
    #N_pts = 300
    #TR = 1
    #A = None
    #N_cluster = 1000
        
    timeline = np.arange(0, TR * N_pts, TR)
    
    h = hrf(timeline)
    H = Hmat(h)
    
    if A == None:
        A = -np.eye(N_nds)    
        A[2,1] = .9
        A[4,1] = .9
        A[3,2] = .9
        A[3,4] = .9
    
    u = np.zeros(shape=(N_pts,N_nds))
    for n in range(N_nds):
        u[:,n],_,_,_ = get_stimulus(TR=TR, duration_range=(1,3))
    
    X = np.zeros(shape=(N_pts,N_nds))
    dt = np.ones(N_nds)
    for t in range(1,N_pts):
        X[t,:] = X[t-1,:] + dt * ( np.dot(A,X[t-1, :]) + u[t, :] )
    
    Y = dict()
    U = dict()
    Y['background'] = np.zeros(shape=(N_pts,))
    U['background'] = np.zeros(shape=(N_pts,))
    for n in range(N_nds):
        key = 'n'+str(n+1)
        s = np.dot(H, X[:,n])
        s = s - s.mean()
        Y[key] = s / s.std()
        U[key] = u[:,n]
    
    if type(N_cluster) == int:
        N = N_cluster
        N_cluster = dict()
        for n in Y.keys():
            N_cluster[n] = N
    
    Y_data = dict(Y)
    U_data = dict(U)
    for i,n in enumerate(Y.keys()):
        Y_data[n] = np.reshape(Y_data[n],(N_pts,1))
        Y_data[n] = np.tile(Y_data[n],(1,N_cluster[n]))
        U_data[n] = np.reshape(U_data[n],(N_pts,1))
        U_data[n] = np.tile(U_data[n],(1,N_cluster[n]))
    
    '''noise extraction'''
    N_pts_noise = 121
    TR_noise = 2.5
    timeline_noise = np.arange(0, TR_noise*N_pts_noise, TR_noise)
    noisedata = nb.load('noise_data/swbold.nii')
    noisedata = np.asarray(noisedata.get_data())
    noisedata[np.isnan(noisedata)] = 0
    noisemask = nb.load('noise_data/msk_noise.nii')
    noisemask = np.asarray(noisemask.get_data())
    noisemask[np.isnan(noisemask)] = 0
    (inoise,jnoise,knoise) = np.where( noisemask == 15 ) # coordinates with noise signals 
    
    w = 1./SNR
    N_noise = len(inoise)
    noise_signals = np.zeros( (N_noise,len(timeline)) )
    for i in range(N_noise):
        noise_signal = noisedata[inoise[i],jnoise[i],knoise[i],:]
        noise_signal = np.interp(timeline, timeline_noise, noise_signal)
        noise_signal = noise_signal - np.mean(noise_signal)
        noise_signals[i,:] = w * (noise_signal / np.std(noise_signal))
    
    #k = 0
    #s = noisedata[inoise[k],jnoise[k],knoise[k],:]
    #s = s - np.mean(s)
    #s = w * (s / np.std(s))
    #plt.plot(timeline_noise,s)
    #plt.plot(timeline, noise_signals[k,:])
      
    '''contamination of signals'''
    for r_label,signals in Y_data.items():
        ns = Y_data[r_label].shape[1]
        for i in range(ns):
            j = np.random.randint(0, N_noise)
            Y_data[r_label][:,i] += noise_signals[j,:]
    
    Y_data['background'] = np.zeros(Y_data['background'].shape)
    ns = Y_data['background'].shape[1]
    for i in range(ns):
        j = np.random.randint(0, N_noise)
        Y_data['background'][:,i] += noise_signals[j,:]

    return Y_data, U_data, H
    
if __name__ == "__main__":

    #probar SNRs = [0.2, 1.0, 4.0]
    Y_data, U_data, H = get_dcm_data(SNR=1)
    Y_mat = np.hstack(Y_data.values())
    U_mat = np.hstack(U_data.values())
    fig, ax = plt.subplots(ncols=2)    
    ax[0].imshow(Y_mat, aspect='auto', interpolation = 'nearest')        
    ax[1].imshow(U_mat, aspect='auto', interpolation = 'nearest')
    plt.tight_layout()
    plt.show()
