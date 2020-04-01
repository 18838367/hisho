import numpy as np
import scipy as scp
import scipy.integrate
from astropy.cosmology import WMAP9 as cosmo, z_at_value
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
import math

def rdiff(z, nu, M, sig):
    rd=2.2e2/((1+z)*nu*M*(sig/100.0)**(0.5))
    return rd

def numberDensity(MHalo, M, projA):
    sigma=MHalo/(M*projA)
    return sigma

def fresnelScale(lam, zL, zS):
    DL=cosmo.angular_diameter_distance(zL).value*1e6
    DS=cosmo.angular_diameter_distance(zS).value*1e6
    DLS=cosmo.angular_diameter_distance_z1z2(zL, zS).value*1e6
    rF=(lam*DL*DLS/(DS*(1+zL)*2.0*math.pi))**(0.5)
    return rF

def tScat(nu, rF, rd):
    t=1.0/(2.0*math.pi*nu)*(rF/rd)**2.0
    return t

def tScatAuto(zL, zS, nu, M, MHalo, rHalo):
    area=math.pi*(rHalo**2)
    sig=numberDensity(MHalo, M, area)
    rd=rdiff(zL, nu, M, sig)
    lam=const.c.value/nu
    rF=fresnelScale(lam/(const.pc.value), zL, zS)
    t=tScat(nu, rF, rd)
    return t

def MScat(zL, zS, nu, t, MHalo, rHalo):
    area=math.pi*(rHalo**2)
    lam=const.c.value/nu    
    rF=fresnelScale(lam/(const.pc.value), zL, zS)
    M=(t*2.0*math.pi*nu*(2.2*10**2)**2*area*100.0)/(rF**2*(1.0+zL)**2*nu**2*MHalo)
    return M

def coherenceArea(nu, zL, zS):
    lam=const.c.value/nu
    cA=math.pi*fresnelScale(lam/const.pc.value, zL, zS)**2
    return cA

def integrand(zL, nu, zS):
    mass=coherenceArea(nu, zL, zS)*cosmo.hubble_distance.value*1e6/((1+zL)*cosmo.efunc(zL))*cosmo.critical_density(zL).value*100**3*const.pc.value**3/(1e3*const.M_sun.value)
    return mass

def integrand2(zL, nu, zS):
    mass=coherenceArea(nu, zL, zS)*cosmo.hubble_distance.value*1e6/((1+zL)*cosmo.efunc(zL))
    return mass

def coherenceMass(zS, nu):
    cN=scipy.integrate.quad(integrand, 0, zS, args=(nu, zS))
    return cN 
