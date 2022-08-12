#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 19:58:18 2021

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# Datos generales de la simulación
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

#%% Acá arranca la simulación

# grilla de sampleo temporal
tt = np.linspace(0, (N-1)*ts, N)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

# Concatenación de matrices:
# guardaremos las señales creadas al ir poblando la siguiente matriz vacía

f0 = np.array([ N/8, N/4 , N*3/8])

plt.close('all')

for ii in f0:
    
    # mute, para ver solo el efecto ventana.
    # xx = np.ones((N,3))
    
    # senoidal    
    xx = np.hstack((
                    np.sin( 2*np.pi*(ii)*df*tt ).reshape(N,1),
                    np.sin( 2*np.pi*(ii+0.25)*df*tt ).reshape(N,1),
                    np.sin( 2*np.pi*(ii+0.5)*df*tt ).reshape(N,1),
                    ))

    # normalizamos en potencia
    xx = xx / np.sqrt(np.mean(xx**2, axis=0))
    
    XX = 1/N*np.fft.fft(xx, axis=0)
    
    bfrec = ff <= fs/2
    
    
    plt.figure()    
    plt.plot(ff[bfrec], 10 * np.log10( 2 * np.abs(XX[bfrec,:])**2 ), ':x' )
    
    plt.ylim((-100, 5))
    
    
