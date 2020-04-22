import numpy as np
import scipy as scp
import scipy.integrate
from astropy.cosmology import Planck18_arXiv_v2 as cosmo, z_at_value
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
import math

#This function set is based on the treatment of gravitational radiation in macquart 2004
#effectively it is to model the temporal signature impressed upon the signal by gravitational 
#scattering from an ensemble of lensing objects

#the diffractive scale for a uniform distribution of compact objects at frequenyc nu, 
#for an ensemble of uniformly distributed mass M lensing
#objects with number density per pc^2 of sigma from a screen at redshift z
#z=redshift, nu=frequency, M=mass (solar masses), sig=number density lensing objects (N/pc^2)
def rdiffUniform(z, nu, M, sig):
    rd=2.2e2/((1+z)*nu*M*(sig/100.0)**(0.5))
    return rd

#diffractive scale for a power law distribution of matter
#deltaL=thickness of scattering medium (kpc), L0=outer scale (kpc)
#beta=index of power law
#Mcell is the mass in each cell, which is technically equivlent to the sqrt(variance of mass)
#simply because if the mass is mostly distributed inhomogeneously then the average is zero
def rdiffPowerLaw(z, nu, Mcell, deltaL, L0, beta):
    rd=1.8e-4*(L0/10.0)**(1.5)/(nu*(1+z)*((beta-3)/beta)**(0.5)*(Mcell/1e11)*(deltaL/5.0)**(0.5))
    return rd

#MHalo=halo mass (solar masses), M=lens mass (solar mass), projA=projected area 
def numberDensity(MHalo, M, projA):
    sigma=MHalo/(M*projA)
    return sigma

#the fresnel scale Rf (pc), lam=the wavelength of radiation (pc), zL=redshift of lens
#zS=redshift of source
def fresnelScale(lam, zL, zS):
    DL=cosmo.angular_diameter_distance(zL).value*1e6
    DS=cosmo.angular_diameter_distance(zS).value*1e6
    DLS=cosmo.angular_diameter_distance_z1z2(zL, zS).value*1e6
    rF=(lam*DL*DLS/(DS*(1+zL)*2.0*math.pi))**(0.5)
    return rF

#gives the scattering timescale
#tScat= scattering timescale (seconds), nu=frequency (Hz), rF (pc), rd (pc)
def tScat(nu, rF, rd):
    t=1.0/(2.0*math.pi*nu)*(rF/rd)**2.0
    return t

#a handleing function that solves for the scattering timescale from base parameter set
#Uses NFW profile with a uniform distribution of dark matter compact objects
#impactP=impact parameter of source from center of galaxy (kpc)
def tScatAutoUniform(zL, zS, nu, M, MHalo, impactP):
    sig=NFWSurfaceDensity(MHalo, impactP, zL)/M/1e6  #number/pc2
    rd=rdiffUniform(zL, nu, M, sig)
    lam=const.c.value/nu
    rF=fresnelScale(lam/(const.pc.value), zL, zS)
    t=tScat(nu, rF, rd)
    return t

#a handling function that solves for the scattering timescale from base parameter set
#Uses an NFW profile with a power law distribution of dark matter (similar to gas)
#Calculates rdiff using the variance of mass in one cell L0**3, if all mass is 
#distributed inhomogenoeusly (as we expect in this model) the variance is equal to 
#the square of the total mass in the cell (Mcell). This model calculates the mass in
#a cell at the radius equal to impact parameter, densist point along LOS and then 
#approximates deltaL (medium thickness) as being equal to the factor that makes 
#Mcell equal to MLOS (the total mass along the line of sight) calculated from the 
#surface density distribution of an NFW profile.
def tScatAutoPowerLaw(zL, zS, nu, MHalo, impactP, L0, beta):
    rho=NFWVolumeDensity(MHalo, impactP, zL)  #Msun/kpc3
    sig=NFWSurfaceDensity(MHalo, impactP, zL) #Msun/kpc2 
    MLOS=sig*L0**2
    Mcell=rho*L0**3
    deltaL=MLOS/Mcell
    print('deltaL : '+str(deltaL) + ' kpc')
    rd=rdiffPowerLaw(zL, nu, Mcell, deltaL, L0, beta) #pc
    lam=const.c.value/nu
    rF=fresnelScale(lam/(const.pc.value), zL, zS) #pc
    print(rd, rF, lam)
    t=tScat(nu, rF, rd)
    return t

#finds the mass of a lens associated with a scattering tail of length t
def MScatUniform(zL, zS, nu, t, MHalo, impactP):
    lam=const.c.value/nu    
    rF=fresnelScale(lam/(const.pc.value), zL, zS)
    Sigma=NFWSurfaceDensity(MHalo, impactP, zL)/1e6
    M=(t*2.0*math.pi*(2.2*10**2)**2*100.0)/(rF**2*(1.0+zL)**2*nu*Sigma)
    return M


#two funcs below are used by NFW
def gFunc(x):
    g=1.0/(np.log(1.0+x)-x/(1.0+x))
    return g


def fFunc(x):
    if (x>1):
        f=1.0-2/((x**2.0-1.0)**(0.5))*np.arctan(((x-1.0)/(x+1.0))**(0.5))
    else:
        if (x<1):
            f=1.0-2/((1.0-x**(2.0))**(0.5))*np.arctanh(((1.0-x)/(1.0+x))**(0.5))
        else:
            f=0.0   
    return f

def f2Func(Rhat, rs, c, r):
    if (r>rs):
        result=(1-abs(c**2*Rhat**2-1)**(-0.5)*np.arccos(1/(Rhat*c)))/((Rhat**2*c**2-1)**(2.0))
    else:
        if (r<rs):
            result=(1-abs(c**2*Rhat**2-1)**(-0.5)*np.arccosh(1/(Rhat*c)))/((Rhat**2*c**2-1)**(2.0))
        else:
            result=1.0/3.0
    return result

def NFWVolumeDensity(virialMass, radius, zL):    #Msun/kpc^3
    rhoCrit=cosmo.critical_density(zL).value/(1000.0*const.M_sun.value)*(100.0*const.pc.value*1000.0)**3.0
    #rhoCrit=critical density (M_sun/kpc^3)
    virialRadius=(virialMass*3.0/(4.0*math.pi*200.0*rhoCrit))**(1.0/3.0) #(kpc)
    c=7.67
    dChar=200.0*c**3*gFunc(c)/3.0
    Rs=virialRadius/c
    rhos=dChar*rhoCrit
    x=radius/Rs
    density=rhos/(x*(1+x)**2.0)
    return density 
    
def NFWSurfaceDensity(virialMass, impactP, zL):  #Msun/kpc^2
    rhoCrit=cosmo.critical_density(zL).value/(1000.0*const.M_sun.value)*(100.0*const.pc.value*1000.0)**3.0
    virialRadius=(virialMass*3.0/(4.0*math.pi*200.0*rhoCrit))**(1.0/3.0)
    c=7.67
    dChar=200.0*c**3*gFunc(c)/3.0
    Rs=virialRadius/c
    rhos=dChar*rhoCrit
    x=impactP/Rs
    Sigma=2*rhos*Rs/(x**2.0-1.0)*fFunc(x)
    return Sigma

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
