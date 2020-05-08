import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
import scipy.special as spec
import scipy.integrate as intg
import math

def equivalentForm(beta, qmin, qmax, r):
    if beta<3:
        hypForm=1
    else:
        if beta>3:
            hypForm=-r**2*qmin**2/4*(2.0+beta)/beta
        else:
            print('not 3 you idiot')
            hypForm=0.0
    return hypForm

def besselInt(b, qmin, qmax, r):
    result=intg.quad(handlerBesselInt, qmin, qmax, args=[r, b])[0]
    return result

def handlerBesselInt(q, args):
    r, b = args
    return spec.jn(0,q*r)*q**(-3-b)

def approxBesselInt(b, q, r):
    return 0.5*q**(-2.0-b)*spec.gamma(-b/2.0-1.0)*spec.hyp1f2(-b/2.0-1.0,1,-b/2.0, q**2*r**2/4.0)[0]/(spec.gamma(1)*spec.gamma(-b/2.0))


def Kvalue(lam, zL):
    return (-8*math.pi*(1.0+zL)/lam*const.G/(const.c**2.0)).value

def Avalue(b, qmin, qmax, sigP):
    if b<3:
        A=4*math.pi*(3.0-b)*qmax**(b-3.0)*sigP**2.0
    else:
        if b>3:
            A=4*math.pi*(3.0-b)*qmin**(b-3.0)*sigP**2.0
        else:
            print('not three')
            A=0
    return A

def dpsi(b, qmin, qmax, r, lam, zL, deltaL, sigP):
    K=Kvalue(lam, zL)
    A=Avalue(b, qmin, qmax, sigP)
    D=4*math.pi*A*deltaL*K**2.0*intg.quad(handlerDpsi, qmin, qmax, args=[r, b])[0]
    return D

def handlerDpsi(q, args):
    r, b = args
    return q**(-3.0-b)*(1.0-spec.jn(0, q*r))

def dpsiApprox(b, qmin, qmax, r, lam, zL, deltaL, sigP):
    K=Kvalue(lam, zL)
    if b<3:
        D=4*math.pi**2.0*K**2.0*deltaL*(3.0-b)*sigP**2.0*r**2.0/(b*qmax**3.0)*(qmax/qmin)**(b)
    else:
        if b>3:
            D=4*math.pi**2.0*K**2.0*deltaL*(3.0-b)*sigP**2.0*r**2.0/(b*qmin**3.0)
        else:
            print('not three')
            D=0
    return D
