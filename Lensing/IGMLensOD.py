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
from LensFunctions import opticalDepth

burstParams=np.loadtxt('burst.params', delimiter=',')
galaxyParams=np.loadtxt('galaxy.params', delimiter=',')
simParams=np.loadtxt('sim.params', delimiter=',')

Mset=np.power(10,np.arange(simParams[0,0],simParams[0,1],simParams[0,2]))
fdm=np.arange(simParams[1,0], simParams[1,1], simParams[1,2])

for i in range(len(burstParams[:,0])):
    tau=opticalDepth(burstParams[i,1], burstParams[i,2], burstParams[i,3], burstParams[i,4], np.amax(galaxyParams[np.where(galaxyParams[:,0]==burstParams[i,0])[0],6]), Mset, fdm, cosmo.Odm0)
    np.save('IGMNLOD'+str(burstParams[i,0]), tau)
