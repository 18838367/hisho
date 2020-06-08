import numpy as np
import matplotlib.pyplot as plt


def rd1(z, nu, Mcell, deltaL, L0, beta, ell0):
    rd=3.6e2*(L0/10.0)**(2.5)/(((3.0-beta)/beta)**(0.5)*(deltaL/5.0)**(0.5)*nu*(1.0+z)*(    Mcell/1e11)*(ell0/0.005)*(L0*1e3/ell0)**(beta/2.0))
    return rd

def rd2(z, nu, Mcell, deltaL, L0, beta, ell0):
    rd=(0.003*10**(beta*3))**(-0.5)*((3-beta)/beta)**(-0.5)*(1+z)**(-1)*nu**(-1)*(Mcell/5e7)**(-    1)*(deltaL/2.0)**(-0.5)*(L0)**((5-beta)/2.0)*(ell0)**((beta-2.0)/2.0)
    return rd
