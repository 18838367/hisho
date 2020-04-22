#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy as scp
import scipy.integrate
from astropy.cosmology import Planck18_arXiv_v2 as cosmo, z_at_value
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
import math
from LensFunctions import nonLensingOD

burstParams=np.loadtxt('burst.params', delimiter=',')
galaxyParams=np.loadtxt('galaxy.params', delimiter=',')
simParams=np.loadtxt("sim.params", delimiter=',')
Nburst=burstParams.shape[0]

#The MACHO masses and dark matter fractions 
Mset=np.power(10,np.arange(simParams[0,0],simParams[0,1],simParams[0,2]))
fdm=np.arange(simParams[1,0], simParams[1,1], simParams[1,2])

#Burst parameters

for j in range(Nburst):
    galacticParams=galaxyParams[np.where(galaxyParams[:,0]==burstParams[j,0])[0],1:]
    if(len(galacticParams.shape)==1):
        contributors=1
    else:
        contributors=galacticParams.shape[0]
    for i in range(contributors):
        #probability of not lensing
        NLOD=nonLensingOD(burstParams[j,1:], galacticParams[i,:], Mset, fdm)
        np.save('GalacticNLOD'+str(burstParams[j,0])+str(i), NLOD)


