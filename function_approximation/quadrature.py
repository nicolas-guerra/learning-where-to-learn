#!/usr/bin/env python
# Credit: Kernel Quadrature with Randomly Pivoted Cholesky by Ethan N. Epperly and Elvira Moreno Ferreira
# https://github.com/eepperly/RPCholesky-Kernel-Quadrature/blob/main/python/quadrature.py

import numpy as np
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
import warnings

def rpcholesky(proposal, num_pts, kernel, algorithm = "reject"):
    pts = np.zeros((num_pts, len(proposal())))
    L = np.zeros((num_pts,num_pts))
    i = 0
    trials = 0
    alpha = 1
    if ("opt" in algorithm):
        warnings.warn("Optimization should only be used when kernel is periodic or domain is all of space", UserWarning)
    while i < num_pts:
        s = proposal()
        d, c, kss = reskernel(s,pts[0:i,:],
                              L[0:i,0:i],
                              kernel)
        trials += 1
        if np.random.rand() < d / kss / alpha:
            pts[i,:] = s
            L[i,0:i] = c
            L[i,i] = np.sqrt(d)
            i += 1
            trials = 0

        if ("opt" in algorithm) and trials >= 25:
            obj_helper = lambda arr: -arr[0] / arr[2]
            objective = lambda s: obj_helper(
                reskernel(s,pts[0:i,:], L[0:i,0:i],
                kernel))
            result = minimize(objective, proposal(),
                              method = "nelder-mead",
                              options = {"xatol":1e-3,
                                         "disp":False})
            trials = 0
            if not result.success:
                warnings.warn("Optimization unsuccessful",RuntimeWarning)
            else:
                alpha = -result.fun

    return pts

def cvs(proposal, num_pts, kernel, num_steps = None):
    pts = np.zeros((num_pts, len(proposal())))
    for i in range(num_pts):
        pts[i,:] = proposal()
    K = kernel(pts, pts)
    detK = np.linalg.det(K)

    if num_steps is None:
        num_steps = 10 * num_pts
    
    for trial in range(num_steps):
        i = np.random.choice(num_pts)
        s = proposal()

        Kprop = np.copy(K)
        Kprop[i,0:i] = kernel(s[np.newaxis,:],
                              pts[0:i,:])
        Kprop[i,i] = kernel(s[np.newaxis,:],
                            s[np.newaxis,:])
        Kprop[i,(i+1):] = kernel(s[np.newaxis,:],
                                 pts[(i+1):,:])
        Kprop[:,i] = Kprop[i,:]

        detKprop = np.linalg.det(Kprop)

        if np.random.rand() < 0.5*min(1,detKprop/detK):
            pts[i,:] = s
            K = Kprop
            detK = detKprop

    return pts

def uniform(proposal, num_pts, kernel=None):
    pts = np.zeros((num_pts, len(proposal())))
    for i in range(num_pts):
        pts[i,:] = proposal()
    return pts

def reskernel(s, pts, L, kernel):
    k = kernel(pts, s[np.newaxis,:])
    c = np.reshape(np.linalg.solve(L, k), (len(pts),))
    kss = float(kernel(s[np.newaxis,:], s[np.newaxis,:]))
    d = kss - np.linalg.norm(c)**2
    return d, c, kss

def weights(pts, integrator=None, kernel=None, total_mass = 1.0, method="optimal"):
    num_pts = pts.shape[0]
    if "optimal" == method:
        integrals = np.zeros(num_pts)
        for i in range(num_pts):
            integrals[i] = integrator(lambda s:
                                      float(kernel(s[:,np.newaxis],
                                                   pts[i,np.newaxis])))
        K = kernel(pts, pts)
        return np.linalg.solve(K + 10*np.finfo(float).eps*np.trace(K)*np.eye(num_pts), integrals)
    elif "uniform" == method:
        return total_mass * np.ones(num_pts) / num_pts
    else:
        raise RuntimeError("Method '{}' unrecognized".format(method))

def integrate(f, pts, wts):
    ans = 0
    for i in range(len(wts)):
        ans += wts[i] * float(f(pts[i,:]))
    return ans
    
if __name__ == "__main__":
    s = 1
    bern = lambda x: x**2 - x + 1/6.0
    prefactor = (-1)**(s-1) * (2*np.pi)**(2*s) / np.math.factorial(2*s)
    sobolev_kernel = lambda X,Y: 1 + prefactor * bern((X - Y.T) % 1)
    
    proposal = lambda: np.random.rand(1)
    integrator = lambda g: 1

    f = lambda x: np.sin(2*np.pi*x)
    print("Simple test of different quadrature methods for integrating f(x) = sin(2*pi*x).")
    print("All methods report the error for a single random realization of the quadrature scheme with 100 nodes.")
    print()

    pts = rpcholesky(proposal, 100, sobolev_kernel)
    wts = weights(pts, integrator, sobolev_kernel)
    print("RPCholesky has error {}".format(np.abs(integrate(f, pts, wts))))

    pts = cvs(proposal, 100, sobolev_kernel)
    wts = weights(pts, integrator, sobolev_kernel)
    print("Continuous volume sampling has error {}".format(np.abs(integrate(f, pts, wts))))

    pts = uniform(proposal, 100, sobolev_kernel)
    wts = weights(pts, method = "uniform")
    print("Monte Carlo has error {}".format(np.abs(integrate(f, pts, wts))))