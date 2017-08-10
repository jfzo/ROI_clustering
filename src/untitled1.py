import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import matplotlib as mpl
from mpltools import style
#style.use('ggplot')
style.use('ieee.transaction')
mpl.rcParams['lines.linewidth'] = 1.5
fontsize = 13

plt.close('all')

def get_stimulus_fcn(N_pts=100, TR=1, onsets=np.arange(11,100,30), duration=1):
    u = np.zeros(N_pts)
    timeline = np.arange(0,N_pts-TR,TR)
    for o in onsets:
        tmp = abs(o - timeline)
        i = tmp.argmin()
        u[ i:i+duration ] = 1
    return u

def hrf(timeline, tau1=5.4, delta1=6., tau2=10.8, delta2=12., c=.35):    
    gamma1 = ((timeline/tau1)**delta1) * np.exp(-(delta1/tau1)*(timeline-tau1))
    gamma2 = ((timeline/tau2)**delta2) * np.exp(-(delta2/tau2)*(timeline-tau2))
    h = gamma1 - c * gamma2
    h = h / h.max()
    return h

def Hmat(h):
    fr = np.zeros( len(h) )
    fr[0] = h[0]
    H = toeplitz(h, fr)
    return H

noisedata = nb.load('noise_data/swbold.nii')
noisedata = np.asarray(noisedata.get_data())
noisedata[np.isnan(noisedata)] = 0
noisemask = nb.load('noise_data/msk_noise.nii')
noisemask = np.asarray(noisemask.get_data())
noisemask[np.isnan(noisemask)] = 0
(inoise,jnoise,knoise) = np.where( noisemask == 15 ) # coordinates with noise signals 

N_pts1 = 121
TR1 = 2.5
timeline1 = np.arange(0, TR1*N_pts1, TR1)
N_pts2 = 300
TR2 = 0.1
timeline2 = np.arange(0, N_pts2+TR2, TR2)
u = get_stimulus_fcn(N_pts=len(timeline2), TR=TR2, onsets=np.arange(25,np.max(timeline2),80), duration=1)

H = Hmat(hrf(timeline2))
Y = np.dot(H,u)
Y = Y - Y.mean()
Y = Y / Y.std()

colors = ['b','r','k','g','b','k','c']
l1,l2 = float('inf'), -float('inf')


SNRs = [1e-4,1,100] #db
SNRs = [0.2,1.0,4.0] #db

SNRs = SNRs[-1::-1]
#SNR = 1

f = plt.figure(1, figsize=(9,3))
axs = [f.add_subplot(131),f.add_subplot(132),f.add_subplot(133)]

for j,SNR in enumerate((SNRs)):
    
    ax = axs[j]

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(fontsize)    
    
    events = [timeline2[k] for k in range( len(u) ) if u[k]==1]
    for e in events:
        ax.axvline(x=e, ymin=0, ymax=1, linewidth=6, color=(0.7,0.7,0.7))
    
    for i,s in enumerate((100,120,140)):
    
        noise_signal = noisedata[inoise[s],jnoise[s],knoise[s],:]
        noise_signal = np.interp(timeline2,timeline1,noise_signal)
        noise_signal = noise_signal - np.mean(noise_signal)
        noise_signal2 = noise_signal / np.std(noise_signal)        
        w = Y.std()/(10**(SNR/20.0))
        w = Y.std()/SNR
        noise_signal2 = w*noise_signal2
        print w,noise_signal.std(), noise_signal2.std()
        Y_plus_noise = Y + noise_signal2
        Y_plus_noise = Y_plus_noise - np.mean(Y_plus_noise)
        Y_plus_noise = Y_plus_noise / np.max(Y_plus_noise)
        ax.plot(timeline2, Y_plus_noise, color=colors[i])#, label='SNR = '+str(SNR))
        l1 = min( min(Y_plus_noise), l1 )
        l2 = max( max(Y_plus_noise), l2 )
    
    #ax.plot(timeline2, Y/Y.max(), color='y', label='Prototype BOLD signal')
    ax.set_title('SNR = '+str(SNR), fontsize=fontsize)    
    ax.set_xlim([np.min(timeline2),np.max(timeline2)])            
    ax.set_ylim([l1,l2])
    #ax.legend(loc='upper left', bbox_to_anchor=(-0.14, 1.035), fancybox=True, shadow=True, ncol=2, fontsize=fontsize)
    ax.xaxis.label.set_color((0,0,0))
    ax.yaxis.label.set_color((0,0,0))
    ax.title.set_color((0,0,0))
    ax.tick_params(axis='x', colors=(0,0,0))
    ax.tick_params(axis='y', colors=(0,0,0))

    if j==0:
        ax.set_ylabel('Amplitude', fontsize=fontsize)
#    if j==len(SNRs)-1:
    ax.set_xlabel('Time (s)', fontsize=fontsize)

plt.tight_layout(pad=0.5,h_pad=0.05, w_pad=0.01)
plt.savefig('exp1_noisy_signals.pdf',dpi=1000)
plt.show() 
 
 