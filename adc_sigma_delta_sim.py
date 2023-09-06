#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 4 16:56:51 2023

Descripción (PDS TS 3/4)
-----------

En este script se simula un conversor analógico digital (ADC)
 mediante la técnica de sobremuestreo más "noise-shaping".
 
 En el script se analiza el efecto 
del ALIAS, producto del muesrtreo, y el ruido de cuantización 
producido por la cuantización.
El experimento permite hacer una pre-distorsión de la "señal 
analógica" simulando un "piso de ruido", luego se analiza cómo
 afecta el mismo a la cuantización del ADC. También se puede 
 analizar si la predistorsión está correlada (mediante una 
señal chirp) o incorrelada (ruido Gaussiano) respecto a la 
senoidal de prueba.

@author: mariano
"""

#%% Configuración e inicio de la simulación

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


# Datos generales de la simulación
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras
R = 10 # realizaciones o experimentos

# cantidad de veces más densa que se supone la grilla temporal para tiempo "continuo"
over_sampling = 16
N_os = N*over_sampling

# Datos del ADC
B = 6 # bits
Vf = 2 # Volts
q = Vf/2**B # Volts

# datos del ruido
kn = 1/10
pot_ruido = q**2/12 * kn # Watts (potencia de la señal 1 W)

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

#%% Acá arranca la simulación

# grilla de sampleo temporal
tt = np.linspace(0, (N-1)*ts, N)
tt_os = np.linspace(0, (N-1)*ts, N_os)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)
ff_os = np.linspace(0, (N-1)*df, N_os)

# Concatenación de matrices:
# guardaremos las señales creadas al ir poblando la siguiente matriz vacía

analog_sig = np.sin( 2*np.pi*(10)*df*tt_os )

analog_sig = analog_sig / np.sqrt(np.var(analog_sig))

# nn2 = np.sin( 2*np.pi*(200)*df*tt_os )

# nn2 = nn2 / np.sqrt(np.var(nn2))

nq = np.zeros((N_os,R))
nn = np.zeros((N_os,R))
sr = np.zeros((N_os,R))
srq = np.zeros((N_os,R))

for rr in range(R):
    
    # Generación de la señal de interferencia
    # incorrelada
    nn[:, rr] = np.random.normal(0, np.sqrt(pot_ruido), size=N_os)
    # nqi = np.random.uniform(low=-q/2, high=q/2, size=N_os)
    
    # muy correlada
    # nn = sig.chirp(tt_os, 2*df, (N-1)*ts, fs/2)
    # nn = nn / np.sqrt(np.var(nn)) * np.sqrt(pot_ruido)
    
    # construimos la señal de entrada al ADC
    sr[:, rr] = analog_sig + nn[:, rr]
    # sr = analog_sig 
    
    #%% Sigma-Delta ADC
    
    wh = np.zeros(N_os)
    vo = np.zeros_like(wh)
    vi = np.zeros_like(wh)
    # pérdidas del integrador
    k = 1
    
    for ii in range(1, N_os):
        
        vi[ii-1] = sr[ii-1, rr] - wh[ii-1]
        
        # copia
        # vo[ii] = vi[ii-1]
    
        #integrador 1º
        vo[ii] = vi[ii-1] + k*vo[ii-1]
    
        #integrador 2º
        # vo[ii] = vi[ii-2] + 2*k*vo[ii-1] - k**2 * vo[ii-2]
    
        # quantizacion
        wh[ii] = q * np.round(vo[ii]/q) 
        # wh[ii] = vo[ii] + nqi[ii] 
        
        
    # muestreo la señal analógica 1 cada OS muestras
    # srq = wh[::over_sampling]
    srq[:, rr] = wh
    
    # 
    # ruido de cuantización
    # nq[:, rr] =  srq - vo[::over_sampling]
    # nq =  nq[::over_sampling]
    nq[:, rr] =  np.roll(srq[:, rr],-1) - sr[:, rr]

#%% Presentación gráfica de los resultados
plt.close('all')

plt.figure(1)
plt.plot(tt_os, srq[:,0], lw=2, linestyle=':', color='blue', marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='ADC out')
plt.plot(tt_os, wh, lw=1, linestyle='--', color='red', marker='o', markersize=3, markerfacecolor='none', markeredgecolor='red', fillstyle='none', label='$ \hat{w} $')
plt.plot(tt_os, analog_sig + nn[:,0], lw=1, color='orange', ls='dotted', label='$ s $ (analog)')
plt.plot(tt_os, vo, lw=1, color='green', ls='--', marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ v_o $ (int. out)')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


plt.figure(2)
ft_SR = 1/N_os*np.fft.fft( sr, axis = 0 )
ft_Srq = 1/N_os*np.fft.fft( srq, axis = 0 )
ft_As = 1/N_os*np.fft.fft( analog_sig, axis = 0)
ft_Nq = 1/N_os*np.fft.fft( nq, axis = 0 )
ft_Nn = 1/N_os*np.fft.fft( nn, axis = 0 )
bfrec = ff <= fs/2
bfrec_os = ff_os <= fs/2

# Nnq_mean = np.mean(np.abs(ft_Nq)**2)
Nnq_mean = np.mean(np.mean(np.abs(ft_Nq)**2, axis=1))
nNn_mean = np.mean(np.mean(np.abs(ft_Nn)**2, axis=1))
# nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Srq)**2, axis=1)[bfrec_os]), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)' )
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.abs(ft_As[bfrec_os])**2), color='orange', ls='dotted', label='$ s $ (analog)' )
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_SR)**2, axis=1)[bfrec_os]), ':g', label='$ s_R = s + n $  (ADC in)' )
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Nn)**2, axis=1)[bfrec_os]), ':r')
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Nq)**2, axis=1)[bfrec_os]), ':c')
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Nnq_mean)) )
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()
# suponiendo valores negativos de potencia ruido en dB
# plt.ylim((1.5*np.min(10* np.log10(2* np.array([Nnq_mean, nNn_mean]))),10))


plt.figure(3)
bins = 10
plt.hist(nq.flatten(), bins=2*bins)
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N_os*R/bins, N_os*R/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))


