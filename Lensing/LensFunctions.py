#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy as scp
import scipy.integrate
from astropy.cosmology import WMAP9 as cosmo, z_at_value
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Rectangle


def timeDelay(M, zd, y):
    G=const.G.value
    c=const.c.value
    Msol=const.M_sun.value
    deltaT=4*G*M*Msol/(c**3)*(1+zd)*((y/2.0*(y**2+4)**(0.5))+np.log(((y**2+4)**(0.5)+y)/((y**2+4)**(0.5)-y)))
    return deltaT


def optimWrapper1(y, M, zd, tCrit):
    return abs(timeDelay(M, zd, y)-tCrit)*1e5


#M in solar masses, distances in Gpc, thetaE in arcseconds
def eRadius(M, Dd, Dds, Ds):
    thetaE=(4*const.G*const.M_sun*M/(const.c**2)*(Dds/(Dd*Ds*10**6*const.pc)))**(0.5)*(180/math.pi*60*60)
    thetaE=(thetaE.value)*u.arcsec
    return thetaE


def magRatio(y):
    return (y**2+2+(y*(y**2+4)**(0.5)))/(y**2+2-(y*(y**2+4)**(0.5)))


def optimWrapper2(y, muCrit):
    return abs(magRatio(y)-muCrit)


def tMag(muCrit):
    y0=1
    optimY=scp.optimize.minimize(optimWrapper2, y0, args=(muCrit), method='L-BFGS-B')
    return optimY.x[0]


#Rf is the reference value for the ratio of magnification between the two lensed images
#ratio of magnification must be greater than this reference value
def ymax(Rf):
    yx=((1+Rf)/(Rf**(0.5))-2)**(0.5)
    return yx


def ymin(M, zd, tCrit): 
    y0=1
    optimY=scp.optimize.minimize(optimWrapper1, y0, args=(M, zd, tCrit), method='L-BFGS-B')
    return optimY.x


def potential(M, r):
    return const.G*M*const.M_sun/(r*u.m)


#M is in units of solar mass
def sRadius(M):
    return 2*const.G.value*M*const.M_sun.value/(const.c.value**2)


def sAngle(M, Dd):
    sRad=sRadius(M)
    return sRad/(Dd*1e6*const.pc)*180/math.pi*60*60


def yminSet(Mset, zd, tCrit):
    count=0
    ysmall=np.zeros(Mset.shape)
    for i in range(Mset.shape[0]):
        for j in range(Mset.shape[1]):
            count=count+1
            if(int(count % ((Mset.shape[0]*Mset.shape[1])/10))==0):
                print(str(count)+"/"+str(Mset.shape[0]*Mset.shape[1]))
            ysmall[i][j]=ymin(Mset[i][j], zd, tCrit)
    return ysmall


def tMuConsistency(deltaT, mu, zs, tol):
    zdset=np.arange(0.01,zs,0.01)
    #Mset=np.power(10,np.arange(0.03,3,0.03))
    Mset=np.arange(1,20,0.2)
    consistency=np.zeros([len(zdset), len(Mset)])
    for i in range(len(zdset)):
        for j in range(len(Mset)):
            consistency[i, j]=ymin(Mset[j], zdset[i], deltaT)-tMag(mu)
    possible=consistency/np.less(abs(consistency), tol)

    fig, ax1 = plt.subplots(1)
    heatmap=ax1.imshow(consistency, cmap='viridis', interpolation='nearest', aspect=len(Mset)/len(zdset))
    possMap=ax1.imshow(abs(possible), cmap='binary', aspect=len(Mset)/len(zdset))
    cbar1=fig.colorbar(heatmap)
    cbar2=fig.colorbar(possMap)
    cbar1.set_label('y$_{\Delta t}$-y$_{\mu}$', fontsize=18)
    cbar2.set_label('|y$_{\Delta t}$-y$_{\mu}$| < 0.01', fontsize=18)
    ax1.set_ylabel("$z_L/z_S$", fontsize=18)
    #ax1.set_xlabel("Lens Mass (log$_{10}$(M$_\odot$))", fontsize=18)
    ax1.set_xlabel("Lens Mass (M$_\odot$)", fontsize=18)
    xlocs=[0,20,40,60,80]
    ylocs=[0, 5, 10, 15, 20, 25, 30, 35, 40]
    ax1.set_xticks(xlocs) 
    #ax1.set_xticklabels(np.around(np.log10(Mset[xlocs]), 2))
    ax1.set_xticklabels(np.round(Mset[xlocs]))
    ax1.set_yticks(ylocs)
    ax1.set_yticklabels(np.around(zdset[ylocs]/zs, 2))
    plt.show()
    fig.savefig('Consistency.png')
    #fig.savefig('ConsistencyLog.png')


def zdExclusionCircle(tMin, tMax, SNR, Sfloor, Dd, Dds, Ds, zd, Mset, plot=False, arcsec=True):
    #Mset=np.power(10,np.arange(0.045,4.5,0.045))
    muMax=SNR/Sfloor
    y_muMax=ymax(muMax)
    #Mset=np.arange(1,20,0.2)
    y_t=np.zeros([len(Mset), 2])
    thetaE=np.zeros(len(Mset))
    for i in range(len(Mset)):
        thetaE[i]=eRadius(Mset[i], Dd, Dds, Ds).value
        thetaStrong=sAngle(Mset[i], Dd).value*50
        y_t[i, 0]=ymin(Mset[i], zd, tMin)
        y_t[i, 1]=ymin(Mset[i], zd, tMax)
        if(y_muMax < y_t[i,1]):
            y_t[i, 1]=y_muMax
        if(y_t[i, 0] < (thetaStrong/thetaE[i])): #strong condition normalised to thetaE units
            y_t[i, 0]= (thetaStrong/thetaE[i])
    
    if(plot):
        fig2, ax2 = plt.subplots(1)
        ax2.plot(Mset, y_t[:,0]*thetaE) #excludes masses between the two lines before they cross
        ax2.plot(Mset, y_t[:,1]*thetaE)
        ax2.set_ylabel("Angular Impact Parameter (arcseconds)", fontsize=18)
        ax2.set_xlabel("Lens Mass (M$_\odot$)", fontsize=18)
        plt.show()
        fig2.savefig("angExclusion.png")
    if(arcsec):
        ret1=y_t[:,0]*thetaE
        ret2=y_t[:,1]*thetaE
    else:
        ret1=y_t[:,0]
        ret2=y_t[:,1]
    return(ret1, ret2)


#def MFPSingularPopulation(Mset, fdm, burstParams, lensParams):
#    #mean free path to lensing from a population of black holes with one mass
#    Mset=np.asarray(Mset)
#    tMin, tMax, SNR, Sfloor=burstParams
#    galaxyMass, galaxyRadius, Dd, Dds, Ds, zd=lensParams
#    galaxyVolume=4.0/3.0*math.pi*galaxyRadius**3
#    totalMM=galaxyMass*fdm
#    numberDensity=totalMM/Mset/galaxyVolume 
#    #print(numberDensity.shape)
#                            #tMin, tMax, SNR, Sfloor, Dd, Dds, Ds, zd, Mset
#    LA, UA=zdExclusionCircle(tMin, tMax, SNR, Sfloor, Dd, Dds, Ds, zd, Mset)
#    Lsig=LA*math.pi/(60*60*180.0)*Dd*1e3
#    Usig=UA*math.pi/(60*60*180.0)*Dd*1e3
#    #print(Usig**2*numberDensity)
#    crossSection=(math.pi*Usig**2)-(math.pi*Lsig**2)
#    crossSection=crossSection*np.greater(crossSection,0)
#    return (1.0/(numberDensity*crossSection))


def NFWGalaxyOD(Mset, fdm, burstParams, lensParams):
    #mean free path to lensing from a population of black holes with one mass
    Mset=np.asarray(Mset)
    tMin, tMax, SNR, Sfloor=burstParams
    virialMass, virialRadius, Dd, Dds, Ds, zd, impactP, traversdedPortion=lensParams
    totalMM=virialMass*fdm
    RNorm=impactP/virialRadius
    c=7.67
    Rs=virialRadius/c
    if impactP>Rs:
        Sigma=c**2.0*gFunc(c)/(2*math.pi)*totalMM/(virialRadius**2.0)*(1-np.abs(c**2.0*RNorm**2.0-1.0)**(-0.5)*np.cos(1.0/(c*RNorm)))/((c**2.0*RNorm**2.0-1.0)**(2.0))*traversedPortion
    else:
        Sigma=c**2.0*gFunc(c)/(2*math.pi)*totalMM/(virialRadius**2.0)*(1-np.abs(c**2.0*RNorm**2.0-1.0)**(-0.5)*np.cosh(1.0/(c*RNorm)))/((c**2.0*RNorm**2.0-1.0)**(2.0))*traversedPortion
    
    #print(numberDensity.shape)
                            #tMin, tMax, SNR, Sfloor, Dd, Dds, Ds, zd, Mset
    LA, UA=zdExclusionCircle(tMin, tMax, SNR, Sfloor, Dd, Dds, Ds, zd, Mset)
    Lsig=LA*math.pi/(60*60*180.0)*Dd*1e3
    Usig=UA*math.pi/(60*60*180.0)*Dd*1e3
    #print(Usig**2*numberDensity)
    crossSection=(math.pi*Usig**2)-(math.pi*Lsig**2)
    crossSection=crossSection*np.greater(crossSection,0)
    OD=Sigma*crossSection
    return OD


def nonLensingProb(burstParams, galaxyParams, Mset, fdm):
    #burstParams[tRes (s), tMax-tStart(s), SNRMax, SNRfloor]
    #galaxyParams[galaxyMass(Msol), galaxyRadius(kpc) Dd(Mpc), Dds(Mpc), Ds(Mpc), redshift lens, traversed(distance travelled through galaxy in Mpc)]
    #Mset=Solar masses (Mass of machos)
    #fdm=MACHO dark matter fraction
    prob=np.zeros([len(Mset),len(fdm)])
    lensParams=galaxyParams[0:len(galaxyParams)-1]
    traversed=galaxyParams[len(galaxyParams)-1]
    for i in range(len(fdm)):
        print(fdm[i])

        MFP=MFPSingularPopulation(Mset, fdm[i], burstParams, lensParams)
        expected=traversed*1e3/MFP
        prob[:,i]=np.exp(-expected)
    return prob

def nonLensingOD(burstParams, galaxyParams, Mset, fdm):
    #burstParams[tRes (s), tMax-tStart(s), SNRMax, SNRfloor]
    #galaxyParams[galaxyMass(Msol), galaxyRadius(kpc) Dd(Mpc), Dds(Mpc), Ds(Mpc), redshift lens, traversed(distance travelled through galaxy in Mpc)]
    #Mset=Solar masses (Mass of machos)
    #fdm=MACHO dark matter fraction
    expected=np.zeros([len(Mset),len(fdm)])
    lensParams=galaxyParams[0:len(galaxyParams)]
    for i in range(len(fdm)):
        print(fdm[i])

        expected[:,i]=NFWGalaxyOD(Mset, fdm[i], burstParams, lensParams)
    return expected


def integrandProbedVolume(zd, tMin, tMax, SNR, Sfloor, zs, Mset):
    Dd=cosmo.angular_diameter_distance(zd)
    Dds=cosmo.angular_diameter_distance_z1z2(zd, zs)
    Ds=cosmo.angular_diameter_distance(zs)
    LA, UA=zdExclusionCircle(tMin, tMax, SNR, Sfloor, Dd, Dds, Ds, zd, Mset)
    Lsig=LA*math.pi/(60*60*180.0)*Dd*1e3
    Usig=UA*math.pi/(60*60*180.0)*Dd*1e3
    #print(Usig**2*numberDensity)
    crossSection=(math.pi*Usig**2)-(math.pi*Lsig**2)
    return crossSection


def probedVolume(tMin, tMax, SNR, Sfloor, zs, Mset):
    volume=np.zeros(len(Mset))
    for i in range(Mset):
        volume[i]=scp.integrate.quad(integrand, 0.01, zs, args=(tMin, tMax, SNR, Sfloor, zs, Mset[i]))
    return volume


def integrandOpticalDepth(zd, tMin, tMax, SNR, Sfloor, zs, Mset):
    Dd=cosmo.angular_diameter_distance(zd)
    Dds=cosmo.angular_diameter_distance_z1z2(zd, zs)
    Ds=cosmo.angular_diameter_distance(zs)
    y_L, y_U=zdExclusionCircle(tMin, tMax, SNR, Sfloor, Dd, Dds, Ds, zd, Mset, plot=False, arcsec=False)
    area=(y_U[0]**2.0-y_L[0]**2.0)
    area=area*np.greater(area,0)
    temp=cosmo.H0**2.0/(cosmo.H(zd)*const.c.value/1e3*u.km/u.s)*Dd*Dds/Ds*(1.0+zd)**2*area
    return temp.value


def opticalDepth(tMin, tMax, SNR, Sfloor, zs, Mset, fdm, OmegaC):
    tau=np.zeros([len(fdm), len(Mset)])
    temp=np.zeros(len(Mset))
    for i in range(len(Mset)):
        temp[i]=3.0/2.0*OmegaC*scp.integrate.quad(integrandOpticalDepth, 0.01, zs-0.01, args=(tMin, tMax, SNR, Sfloor, zs, [Mset[i]]))[0]
    tau=np.repeat(np.expand_dims(temp,1), len(fdm), 1).T*np.repeat(np.expand_dims(fdm, 1), len(Mset), 1)
    return tau


#deltaT in seconds, tol in seconds, betaMax in arcseconds
def fixedZDelT(deltaT, tol, betaMax, betaRes, z, zd, rf, tCrit, plot=True):
    """Plots the allowed time delays from the strong gravitational lensing of a lens of mass x lensing a source at
        relative angular impact parameter y 
        deltaT=observed time delay
        tol= tolerance of time delays
        betaMax= maximum angular impact parameter of source
        betaRes= array resolution of tested angular impact params
        z= redshift of source
        zd= redshift of lens
        rf= ratio of SNR of image peaks      
        plot= to plot or not to plot"""
    
    Ds=cosmo.angular_diameter_distance(z)
    Dd=cosmo.angular_diameter_distance(zd) #calculate the relevant angular diameter distances
    Dds=cosmo.angular_diameter_distance_z1z2(zd, z)
    betaSet=np.arange(betaRes, betaMax, betaRes)   #the set of angular impact parameters being tested in arcseconds
    Mset=np.arange(0,100,1)  #the set of mass parameters being tested
    paramSet=np.zeros([len(betaSet), len(Mset), 2]) #combined test parameter array
    paramSet[:,:,0]=np.repeat(np.expand_dims(betaSet,1),len(Mset), 1)  #this process just sets up the array for the map
    paramSet[:,:,1]=np.repeat(np.expand_dims(Mset,1),len(betaSet), 1).T
    thetaE=np.zeros([len(betaSet), len(Mset)])
    thetaE=eRadius(paramSet[:, :, 1], Dd, Dds, Ds).value #calculates the einstein radius 
                                                                              #in arcseconds 
    physicalSize=2*const.G.value*paramSet[:,:,1]/(const.c.value**2) #schwarschild radius of the lens (black hole)
    betaPhysical=physicalSize/Dd.value #angular size of the lens
    yPhysical=betaPhysical/thetaE  #normalised size of the lens
    y=paramSet[:,:,0]/thetaE  #normalised impact parameter of the source object
    magniAllowed=np.less(y,ymax(rf)) #boolean array of impact params less than allowed max based on magnification
    physicalAllowed=np.greater(y,yPhysical) #boolean array of impact params large enough to not be physically obscured 
                                                #by the lens itself
    resAllowed=np.greater(y,yminSet(paramSet[:,:,1], zd, tCrit)) #boolean array of impact params greater than allowed min based on temporal signal width
    #resAllowed=np.greater(y,0)
    timeset=np.zeros(len(Mset))
    timeset=timeDelay(Mset,zd,y) #timeDelays between two lens manifested images 
    closeT=np.where(abs(timeset-deltaT)<tol) #boolean array of timeDelays close to the observed delay
    if(plot):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey='all')
        heatmap=ax1.imshow(timeset, cmap='hot', interpolation='nearest', aspect=len(Mset)/len(betaSet))
        cbar=fig.colorbar(heatmap)
        ax1.set_ylabel("Angular Impact Parameter of Source (microarcseconds)")
        ax1.set_xlabel("Lens Mass (M$_\odot$)")
        ax1.set_yticklabels([0,0,0.25,0.50,0.75,1.0,1.25,1.50,1.75,2.0]) #times by resAllowed when uve done numerical implementation
        positions=ax2.imshow(np.greater(tol,abs(timeset-deltaT))*magniAllowed*physicalAllowed*resAllowed, cmap='hot', interpolation='nearest', aspect=len(Mset)/len(betaSet))
        ax2.set_xlabel("Lens Mass (M$_\odot$)")
    
   # for i in range(np.shape(closeT)[1]):
   #     ax1.add_patch(Rectangle((closeT[0][i], closeT[1][i]), 1, 1, fill=False, edgecolor='blue', lw=0.5))
        #ax1.scatter(closeT[1][:], closeT[0][:], color="blue", s=4)
        plt.show()
        fig.savefig('ImpactVsMass.png')
    return closeT

def zdECShowStrong(tMin, tMax, SNR, Sfloor, Dd, Dds, Ds, zd, Mset, plot=False, arcsec=True):
    #Mset=np.power(10,np.arange(0.045,4.5,0.045))
    muMax=SNR/Sfloor
    y_muMax=ymax(muMax)
    #Mset=np.arange(1,20,0.2)
    y_t=np.zeros([len(Mset), 2])
    y_Field=np.zeros(len(Mset))
    thetaE=np.zeros(len(Mset))
    for i in range(len(Mset)):
        thetaE[i]=eRadius(Mset[i], Dd, Dds, Ds).value
        thetaStrong=sAngle(Mset[i], Dd).value*50
        y_t[i, 0]=ymin(Mset[i], zd, tMin)
        y_t[i, 1]=ymin(Mset[i], zd, tMax)
        if(y_muMax < y_t[i,1]):
            y_t[i, 1]=y_muMax
        y_Field[i]=(thetaStrong/thetaE[i])
        if(y_t[i, 0] < (thetaStrong/thetaE[i])): #strong condition normalised to thetaE units
            y_t[i, 0]= (thetaStrong/thetaE[i])
    
    if(plot):
        fig2, ax2 = plt.subplots(1)
        ax2.plot(Mset, y_t[:,0]*thetaE) #excludes masses between the two lines before they cross
        ax2.plot(Mset, y_t[:,1]*thetaE)
        ax2.set_ylabel("Angular Impact Parameter (arcseconds)", fontsize=18)
        ax2.set_xlabel("Lens Mass (M$_\odot$)", fontsize=18)
        plt.show()
        fig2.savefig("angExclusion.png")
    if(arcsec):
        ret1=y_t[:,0]*thetaE
        ret2=y_t[:,1]*thetaE
        ret3=y_Field*thetaE
    else:
        ret1=y_t[:,0]
        ret2=y_t[:,1]
        ret3=y_Field
    return(ret1, ret2, ret3)
