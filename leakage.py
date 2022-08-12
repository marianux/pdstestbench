#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:56:51 2021

PDS TS 5


@author: mariano
"""

#%% Configuración e inicio de la simulación

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


# Datos generales de la simulación
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

# Cantidad de zero-padding, expande la secuencia con (cant_pad-1) ceros.
cant_pad = 10

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
    
    # zero padding
    zz = np.zeros_like(xx)
    
    # otro padd
    # xx = xx.repeat(10, axis = 0)
    
    # Potencia unitaria
    xx = xx / np.sqrt(np.var(xx, axis=0))

    # padded N
    xx_pad = np.vstack((xx, zz.repeat((cant_pad-1), axis=0)))
    N_pad = xx_pad.shape[0]    
    df_pad = fs/N_pad # resolución espectral
    ff_pad = np.linspace(0, (N_pad-1)*df_pad, N_pad)

    # Potencia unitaria
    # xx_pad = xx_pad / np.sqrt(np.var(xx_pad, axis=0))
    
    # Max. PSD a 0 dB (unitario)
    # El kernel de Dirich. tiene un valor de (N/N_pad) en su lóbulo central.
    # entonces lo compensamos para que de 1.
    xx_pad = xx_pad * N_pad / N

    # # Energía unitaria
    # xx = xx / np.sqrt(np.sum(xx**2, axis=0))
    # xx_pad = xx_pad / np.sqrt(np.sum(xx_pad**2, axis=0))
    
    #%% Presentación gráfica de los resultados
    
    plt.figure()
    ft_XX_pad = 1/N_pad*np.fft.fft( xx_pad, axis = 0 )
    ft_XX = 1/N*np.fft.fft( xx, axis = 0 )
    bfrec = ff <= fs/2
    bfrec_pad = ff_pad <= fs/2
    
    # Potencia total
    xx_pot = np.sum(np.abs(ft_XX)**2, axis = 0)
    xx_pot_pad = np.sum(np.abs(ft_XX_pad)**2, axis = 0)
    
    # ventana duplicadora
    ww = np.vstack((1, 2*np.ones((N//2-1,1)) ,1))
    ww_pad = np.vstack((1, 2*np.ones((N_pad//2-1,1)) ,1))
    
    plt.plot( ff[bfrec], 10* np.log10(ww * np.abs(ft_XX[bfrec,:])**2 + 1e-10), ls='dotted', marker='o' )
    plt.plot( ff_pad[bfrec_pad], 10* np.log10(ww_pad * np.abs(ft_XX_pad[bfrec_pad,:])**2 + 1e-10), ls='dotted', marker='x' )
     
    # plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
    plt.ylabel('Densidad de Potencia [dB]')
    plt.xlabel('Frecuencia [Hz]')
    plt.title('PSD de una senoidal con diferentes desintonías')
    axes_hdl = plt.gca()
    axes_hdl.legend([
        'No leak $\sigma^2 = {:3.3f}$'.format(xx_pot[0]), 
        'mid leak $\sigma^2 = {:3.3f}$'.format(xx_pot[1]), 
        'max leak $\sigma^2 = {:3.3f}$'.format(xx_pot[2]),
        'No leak $\sigma^2 = {:3.3f}$'.format(xx_pot_pad[0]), 
        'mid leak $\sigma^2 = {:3.3f}$'.format(xx_pot_pad[1]), 
        'max leak $\sigma^2 = {:3.3f}$'.format(xx_pot_pad[2])
        ])
    # suponiendo valores negativos de potencia ruido en dB
    plt.ylim((-80, 5))


