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


def blackman_tukey(x,  M = None):    
    
    # N = len(x)
    x_z = x.shape
    
    N = np.max(x_z)
    
    if M is None:
        M = N//5
    
    r_len = 2*M-1

    # hay que aplanar los arrays por np.correlate.
    # usaremos el modo same que simplifica el tratamiento
    # de la autocorr
    xx = x.ravel()[:r_len];

    r = np.correlate(xx, xx, mode='same') / r_len

    Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )

    Px = Px.reshape(x_z)

    return Px;


# Datos generales de la simulación
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

# Cantidad de zero-padding, expande la secuencia con (cant_pad-1) ceros.
cant_pad = 1

#%% Acá arranca la simulación

# grilla de sampleo temporal
tt = np.linspace(0, (N-1)*ts, N)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

# Concatenación de matrices:
# guardaremos las señales creadas al ir poblando la siguiente matriz vacía

# f0 = np.array([ N/8, N/4 , N*3/8])
f0 = np.array([  N/4 ])

plt.close('all')

for ii in f0:
    
    # mute, para ver solo el efecto ventana.
    # xx = np.ones((N,3))
    
    # senoidal    
    xx = np.sin( 2*np.pi*(ii)*df*tt ).reshape(N,1) + \
         np.sin( 2*np.pi*(ii+5)*df*tt ).reshape(N,1)
    
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
    xx_pad = xx_pad / np.sqrt(np.var(xx_pad, axis=0))
    
    # Max. PSD a 0 dB (unitario)
    # El kernel de Dirich. tiene un valor de (N/N_pad) en su lóbulo central.
    # entonces lo compensamos para que de 1.
    # xx_pad = xx_pad * N_pad / N

    # # Energía unitaria
    # xx = xx / np.sqrt(np.sum(xx**2, axis=0))
    # xx_pad = xx_pad / np.sqrt(np.sum(xx_pad**2, axis=0))
    
    #%% Presentación gráfica de los resultados
    
    plt.figure()
    ft_XX_pdg = 1/N_pad*np.fft.fft( xx, axis = 0 )
    ft_XX_bt = blackman_tukey( xx, N//5 )
    ff_wl, ft_XX_wl = sig.welch( xx, nperseg = N//5, axis = 0 )
    
    bfrec = ff <= fs/2
    bfrec_pad = ff_pad <= fs/2
    
    # Potencia total
    xx_pot_pdg = np.sum(np.abs(ft_XX_pdg)**2, axis = 0)
    xx_pot_bt = np.sum(np.abs(ft_XX_bt)**2, axis = 0)
    xx_pot_wl = np.sum(np.abs(ft_XX_wl)**2, axis = 0)
    
    # ventana duplicadora
    ww = np.vstack((1, 2*np.ones((N//2-1,1)) ,1))
    ww_pad = np.vstack((1, 2*np.ones((N_pad//2-1,1)) ,1))
    
    plt.plot( ff[bfrec], 10* np.log10(ww * np.abs(ft_XX_pdg[bfrec,:])**2 + 1e-10), ls='dotted', marker='o', label = f'Per. $\sigma^2 = $ {xx_pot_pdg[0]:3.3}' )
    plt.plot( ff[bfrec], 10* np.log10(ww * np.abs(ft_XX_bt[bfrec,:])**2 + 1e-10), ls='dotted', marker='o', label = f'BT $\sigma^2 = $ {xx_pot_bt[0]:3.3}' )
    plt.plot( ff_wl * fs,     10* np.log10( np.abs(ft_XX_wl)**2 + 1e-10), ls='dotted', marker='o', label = f'Welch $\sigma^2 = $ {xx_pot_wl[0]:3.3}' )
    # plt.plot( ff_pad[bfrec_pad], 10* np.log10(ww_pad * np.abs(ft_XX_pdg[bfrec_pad,:])**2 + 1e-10), ls='dotted', marker='x', label = f'$\sigma^2 = $ {xx_pot[0]:3.3}'  )
     
    # plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
    plt.ylabel('Densidad de Potencia [dB]')
    plt.xlabel('Frecuencia [Hz]')
    plt.title('PSD de una senoidal con diferentes desintonías')
    axes_hdl = plt.gca()
    axes_hdl.legend()

    # suponiendo valores negativos de potencia ruido en dB
    # plt.ylim((-80, 5))


