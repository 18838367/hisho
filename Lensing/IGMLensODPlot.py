import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams['font.family']='serif'
plt.rcParams.update({'font.size':20})
plt.rcParams['font.serif']=['Times New Roman'] 
plt.rcParams['font.serif']


simParams=np.loadtxt('sim.params', delimiter=',')
Mset=np.power(10,np.arange(simParams[0,0],simParams[0,1], simParams[0,2]))
fdm=np.arange(simParams[1,0], simParams[1,1], simParams[1,2])
burstParams=np.loadtxt('burst.params', delimiter=',')
NLODTotal=0

for i in range(len(burstParams[:,0])):
    NLOD=np.load('IGMNLOD'+str(burstParams[i,0])+'.npy')
    NLODTotal=NLOD+NLODTotal
    fig, ax1 = plt.subplots(1)
    heatmap=ax1.imshow(NLOD, cmap='viridis', interpolation='nearest', aspect=len(Mset)/len(fdm)-3, origin='lower')
    cbar1=fig.colorbar(heatmap)
    cbar1.set_label(r'$\tau$', fontsize=28)
    ax1.set_ylabel("f$_{DM}$ (%)", fontsize=28)
    ax1.set_xlabel("Lens Mass (log$_{10}$(M$_\odot$))", fontsize=28)
    #ax1.set_xlabel("Lens Mass (M$_\odot$)", fontsize=18)
    xlocs=[0, 99, 199, 299, 399, 499, 599]
    ylocs=[0, 19, 39, 59, 79]
    ax1.set_xticks(xlocs) 
    ax1.set_xticklabels(np.around(np.log10(Mset[xlocs])+0.01, 1))
    #ax1.set_xticklabels(np.round(Mset[xlocs]))
    ax1.set_yticks(ylocs)
    ax1.set_yticklabels(np.around(fdm[ylocs], 2))
    plt.show()
    fig.savefig('IGMLensOD'+str(burstParams[i,0])+'.pdf')
    np.save('IGMODTotal'+str(burstParams[i,0])+'.npy', NLOD)


fig, ax1 = plt.subplots(1)
heatmap=ax1.imshow(NLODTotal, cmap='viridis', interpolation='nearest', aspect=len(Mset)/len(fdm)-3, origin='lower')
np.save('IGMODTotal.npy', NLODTotal)
cbar1=fig.colorbar(heatmap)
cbar1.set_label(r'$\tau$', fontsize=28)
#ax1.axvline(119, linestyle="--", color='white')
#ax1.axvline(128, linestyle="--", color='white')
#ax1.text(0,105, r'$\sigma_{181112}=0$')
#ax1.text(121,105, r'$\sigma_{180924}=0$')
ax1.set_ylabel("f$_{DM}$ (%)", fontsize=28)
ax1.set_xlabel("Lens Mass (log$_{10}$(M$_\odot$))", fontsize=28)
#ax1.set_xlabel("Lens Mass (M$_\odot$)", fontsize=18)
xlocs=[0, 99, 199, 299, 399, 499, 599]
ylocs=[0, 19, 39, 59, 79]
ax1.set_xticks(xlocs) 
ax1.set_xticklabels(np.around(np.log10(Mset[xlocs])+0.01, 1))
#ax1.set_xticklabels(np.round(Mset[xlocs]))
ax1.set_yticks(ylocs)
ax1.set_yticklabels(np.around(fdm[ylocs], 2))
plt.show()
fig.savefig('IGMLensODTotal.pdf')
