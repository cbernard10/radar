# -*- coding: utf-8 -*-

import numpy as np

def takagi_dec(Z,tol):
    
    if (type(Z) == complex):
        Z = np.array([Z])
        print(Z)
    
    n = Z.shape[0]

    x0 = np.random.rand(n)
    x0 /= np.linalg.norm(x0)
    
    y = []
    x = [x0]
    l = []
    s = []
    v = []
    A = Z.copy()
    
    takagi_vals = []
    takagi_vects = []
    
    cond = True
    
    for k in range(n):
        
        while(cond):
            
            y_ = A @ np.conj(x[-1])
            x_ = y_ / np.linalg.norm(y_)
            
            l_ = np.conj(x_.T) @ A @ np.conj(x_)
            
            s_ = np.absolute(l_)
            v_ = np.exp(1j*np.angle(l_)/2) * x_
            
            y.append(y_)
            x.append(x_)
            l.append(l_)
            s.append(s_)
            v.append(v_)
         
            val = np.linalg.norm(A @ np.conj(v_) - s_*v_) / np.linalg.norm(np.linalg.det(A) * np.linalg.norm(v_) + np.linalg.norm(s_)*np.linalg.norm(v_))
            
            cond = val > tol
        
        takagi_vals.append(s[-1])
        takagi_vects.append(v[-1])
        
        x0 = np.random.rand(n)
        x0 /= np.linalg.norm(x0)
        x = [x0]
        y = []
        l = []
        s = []
        v = []
        cond = True
        A = (np.identity(n) - np.outer(v_, np.conj(v_))) @ A @ (np.identity(n) - np.outer(v_, np.conj(v_))).T

    P = np.diag(takagi_vals)
    U = np.stack(takagi_vects,axis=1)
        
    return U,P

#%%

def takagi_dec1d(z):

    a,b = np.real(z),np.imag(z)   
    u2 = (a+1j*b) / np.absolute(z)    
    u = np.sqrt(u2)    
    p = np.real(z/u2)
    
    return u,p   

