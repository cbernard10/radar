# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from radarpkg2.processing.takagi import takagi_dec1d

def show_subplots(rows,cols,data,plt_type='plot',interpolation='nearest',axis='on',extent=None,X=None,Y=None,titles=None,vmin=None,vmax=None,cmap=None):

    """ affiche les éléments de data dans l'ordre sur rows lignes et cols colonnes """

    fig = plt.figure()
    for i,d in enumerate(data):
        if plt_type=='plot':
            ax = fig.add_subplot(rows,cols,i+1)
            ax.plot(d)
        if plt_type=='imshow':
            ax = fig.add_subplot(rows,cols,i+1)
#            if titles:
#                if len(titles) == len(data):
#                    ax.set_title(titles[i])
            if extent:
                ax.imshow(d,cmap=cmap,vmin=vmin,vmax=vmax,interpolation=interpolation,extent=extent)
            else:
                ax.imshow(d,cmap=cmap,vmin=vmin,vmax=vmax,interpolation=interpolation)
            ax.axis(axis)
        if plt_type=='surface':
            ax = fig.add_subplot(rows,cols,i+1,projection='3d')
            ax.plot_surface(X,Y,d)
        if plt_type=='wireframe':
            ax = fig.add_subplot(rows,cols,i+1,projection='3d')
            ax.plot_wireframe(X,Y,d)

def length(L):

    if isinstance(L,np.ndarray) or isinstance(L,list):
        return len(L)
    else:
        return 1

def Sym_gen():

    a = np.random.rand()-0.5
    return a

def SDP_gen():

    a = np.random.rand()
    return a**2

def plane_to_disk(z):

    y = (z-1j) * 1/(z+1j)

    return y

def disk_to_plane(y):

    z = -1j* 1/(y-1) * (y+1)

    return z

def Siegel_gen(size=1):

    if size==1:
        return Sym_gen() + 1j*SDP_gen()
    L=[]
    for i in range(size):
        L.append(Sym_gen() + 1j*SDP_gen())
    return L

def disk_gen(size=1):

    if size==1:
        return plane_to_disk(Siegel_gen())
    L=[]
    for i in range(size):
        L.append(plane_to_disk(Siegel_gen()))

    return np.array(L)

def action(gZ,z):

    """ si Z génère g, alors g : 0 |--> Z """

    a,b,c,d = gZ

    return ((a*z + b) * 1/(c*z + d))

def symplectic_ker_gen():

    return np.array([[0,1],
                     [-1,0]])

def symplectic_inverse(gZ):

    gZ_1 = np.zeros_like(gZ)

    gZ_1[0,:,:] = gZ[3,:,:]
    gZ_1[1,:,:] = -gZ[2,:,:]
    gZ_1[2,:,:] = -gZ[1,:,:]
    gZ_1[3,:,:] = gZ[0,:,:]

    return gZ_1

def is_symplectic(gz):

    """ dit si gz est symplectique """

    J = symplectic_ker_gen()

    return np.allclose(J, gz.T @ J @ gz)

def make_gZ(u,p):

    """ Z = U @ P @ U.T € D_n
        output : gZ symplectique, gZ : 0_n |--> Z """

    tau = np.arctanh(p)

    a0 = np.cosh(tau)
    b0 = np.sinh(tau)

    n = length(p)

    gZ = np.zeros((4,n,n),dtype=np.complex128)

    """ construction de la forme polaire """

    gZ[0,:,:] = u*a0
    gZ[1,:,:] = u*b0
    gZ[2,:,:] = np.conj(u)*b0
    gZ[3,:,:] = np.conj(u)*a0

    return gZ

def dist(z1,z2):

    """ entre deux points dans le disque de Poincaré """

    a = disk_to_plane(z1)
    b = disk_to_plane(z2)

    birapport = (a-b)/(a - np.conj(b))*(np.conj(a) - np.conj(b))/(np.conj(a) - b)

    eigs = birapport

    return np.real(np.sqrt(np.log((1+np.sqrt(eigs))/(1-np.sqrt(eigs)))**2))

def dist2(z1,z2):
    
    Z1 = np.complex128(np.complex(z1[0],z1[1]))
    Z2 = np.complex128(np.complex(z2[0],z2[1]))

    return dist(Z1,Z2)        

def theta_0(p):

    """ P valeurs de Takagi de Z """

    tau = np.arctanh(p)

    return np.sinh(2*tau)/(2*tau)

def theta(z1,z2):

    u1,p1 = takagi_dec1d(z1)
    u2,p2 = takagi_dec1d(z2)

    gz1 = symplectic_inverse(make_gZ(u1,p1))

    Z = action(gz1,z2)

    return theta_0(takagi_dec1d(Z)[1])

#%%

#z = 0.9j
#
#gZ = make_gZ(*takagi_dec1d(z)).reshape((4))
#
#xs = gZ.real
#ys = gZ.imag
#
#plt.scatter([z.real],[z.imag],c='k')
#plt.scatter(xs,ys,c=['b','g','y','r'])
#
#C = plt.Circle((0,0),1,color='black',fill=0)
#
#ax = plt.gca()
#ax.add_artist(C)
#ax.set_xlim([-2,2])
#ax.set_ylim([-2,2])
#ax.set_aspect('equal')


