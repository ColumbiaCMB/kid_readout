import numpy as np
import scipy.integrate

def trapz2(func, a, b, eps, npoints, args):
    eps = (b-a)*eps
    midpoint = 0.5*(a+b)
    xpoints = a + np.logspace(np.log10(eps),np.log10(midpoint-a),npoints/2)
    ypoints = func(xpoints, *args)
    part1 = scipy.integrate.trapz(ypoints,xpoints)

    xpoints = b - np.logspace(np.log10(eps), np.log10(b-midpoint), npoints/2)
    ypoints = func(xpoints,*args)
    part2 = -1*scipy.integrate.trapz(ypoints,xpoints)
    return part1 + part2

def mb_integrand_1a(eps,delta,kbt,hf):
    f2 = 1/(np.exp((eps+hf)/kbt)+1)
    return ((1-2*f2) * np.abs(eps * (eps+hf) + delta**2)/
            (np.sqrt(eps**2-delta**2)*np.sqrt((eps+hf)**2-delta**2)))

def mb_integrand_1b(eps, delta, kbt, hf):
    f1 = 1/(np.exp(eps/kbt)+1)
    f2 = 1/(np.exp((eps+hf)/kbt)+1)
    return ((f1-f2) * (eps * (eps+hf) + delta**2)/
            (np.sqrt(eps**2-delta**2)*np.sqrt((eps+hf)**2-delta**2)))

def _mb1(delta, kbt, hf):
    s1b = 2*trapz2(mb_integrand_1b, delta, 20*delta, 1e-13, 1000, args=(delta,kbt,hf))/hf
    if hf > 2*delta:
        s1a = trapz2(mb_integrand_1a, delta-hf, -delta, 1e-13, 1000, args = (delta,kbt,hf))/hf
    else:
        s1a = 0
    return s1a + s1b

def mb_integrand_2(eps,delta,kbt,hf):
    f2 = 1/(np.exp((eps+hf)/kbt)+1)
    return ((1-2*f2) * np.abs(eps * (eps+hf) + delta**2)/
            (np.sqrt(delta**2-eps**2)*np.sqrt((eps+hf)**2-delta**2)))

def _mb2(delta, kbt, hf):
    return trapz2(mb_integrand_2, delta - hf, delta, 1e-10, 6000, args=(delta,kbt,hf))/hf