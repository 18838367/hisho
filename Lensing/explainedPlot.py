import numpy as np
import scipy as scp
import scipy.integrate
from astropy.cosmology import WMAP9 as cosmo, z_at_value
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
import math
from LensFunctions import nonLensingProb
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams['font.family']='serif'
plt.rcParams.update({'font.size':20})
plt.rcParams['font.serif']=['Times New Roman'] 
plt.rcParams['font.serif']

simParams=np.loadtxt('sim.params', delimiter=',')
Mset=np.power(10, np.arange(simParams[0,0], simParams[0,1], simParams[0,2]))[100:]
fdm=np.arange(simParams[1,0], simParams[1,1], simParams[1,2])
galaxyParams=np.loadtxt('galaxy.params', delimiter=',')
burstParams=np.loadtxt('burst.params', delimiter=',')
NLprob=1.0
for j in range(len(np.where(galaxyParams[:,0]==burstParams[0,0]))):    
    NLprob=NLprob*np.load('GalacticNLprob'+str(burstParams[0,0])+str(j)+'.npy')[100:,:]
#plot of lensing probability (1-prob of not lensing)
fig, ax1 = plt.subplots(1)
heatmap=ax1.imshow(1-NLprob.T, cmap='viridis', interpolation='nearest', aspect=len(Mset)/len(fdm)-3, origin='lower')
cbar1=fig.colorbar(heatmap)
cbar1.set_label('P$_L$', fontsize=28)
ax1.set_ylabel("F$_{DM}$ (%)", fontsize=28)
ax1.set_xlabel("Lens Mass (log$_{10}$(M$_\odot$))", fontsize=28)
#ax1.set_xlabel("Lens Mass (M$_\odot$)", fontsize=18)
ax1.axvline(491, linestyle="--", color='black')
ax1.axvline(34, linestyle="--", color='white')
ax1.text(500,108, "Delay", fontsize=28)
ax1.text(500,100, "Limited", fontsize=28)
ax1.text(160,104, "Magnification Limited", fontsize=28)
ax1.text(13,104, r'$\sigma=0$')
xlocs=[0, 99, 199, 299, 399, 499]
ylocs=[0, 19, 39, 59, 79]
ax1.set_xticks(xlocs) 
ax1.set_xticklabels(np.around(np.log10(Mset[xlocs])+0.01, 1))
#ax1.set_xticklabels(np.round(Mset[xlocs]))
ax1.set_yticks(ylocs)
ax1.set_yticklabels(np.around(fdm[ylocs], 2))
plt.show()
fig.savefig('explainedPlot.pdf')
