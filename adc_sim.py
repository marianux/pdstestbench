#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:56:51 2021

Descripción (PDS TS 3/4)
-----------

En este script se simula un conversor analógico digital (ADC)
 mediante el sobremuestreo y cuantización de una señal más 
 densamente muestreada, simulando una señal "continua"
en tiempo y amplitud. En el script se analiza el efecto 
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
over_sampling = 32
N_os = N*over_sampling

# Datos del ADC
B = 16 # bits
Vf = 2 # rango simétrico de +/- Vf Volts
q = Vf/2**(B-1) # paso de cuantización de q Volts

# datos del ruido
kn = 1/5
pot_ruido = q**2/12 * kn # Watts (potencia de la señal 1 W)

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

#%% Diseño del filtro de diezmado

# FIR Tipo 2 fuerzo cant_coef par
cant_coef = 500
antisymmetric = False

fpass = 1/over_sampling # 
fstop = np.min([fpass+0.15, 0.8]) # Hz
ripple = 0.5 # dB
attenuation = 40 # dB

# pasa bajo
frecs = [0.0,  fpass,       fstop,         1.0]
# la mitad de la att porque usaremos filtfilt
gains = [0,   -ripple/2, -attenuation/2,  -np.inf] # dB

gains = 10**(np.array(gains)/20)

# algunas ventanas para evaluar
#win_name = 'boxcar'
#win_name = 'hamming'
win_name = 'blackmanharris'
#win_name = 'flattop'


# FIR design
num = sig.firwin2(cant_coef, frecs, gains, window=win_name, antisymmetric=antisymmetric  )
den = 1.0


# Análisis del filtro
# wrad, hh = sig.freqz(num, den)
# ww = wrad / np.pi

# plt.figure(3)

# plt.plot(ww, 20 * np.log10(abs(hh)))

# plot_plantilla(filter_type = 'lowpass' , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)

# plt.title('FIR designed by window method')
# plt.xlabel('Frequencia normalizada')
# plt.ylabel('Modulo [dB]')
# plt.grid(which='both', axis='both')

# axes_hdl = plt.gca()
# axes_hdl.legend()

# plt.show()

#%% Acá arranca la simulación

# grilla de sampleo temporal
tt = np.linspace(0, (N-1)*ts, N)
tt_os = np.linspace(0, (N-1)*ts, N_os)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)
ff_os = np.linspace(0, (N_os-1)*df, N_os)

# Concatenación de matrices:
# guardaremos las señales creadas al ir poblando la siguiente matriz vacía

analog_sig = np.sin( 2*np.pi*250*df*tt_os )

# normalizo en potencia
analog_sig = analog_sig / np.sqrt(np.var(analog_sig))

nq = np.zeros((N_os,R))
nn = np.zeros((N_os,R))
sr = np.zeros((N_os,R))
srq = np.zeros((N_os,R))

for rr in range(R):

    # Generación de la señal de interferencia
    # incorrelada
    nn[:, rr]  = np.random.normal(0, np.sqrt(pot_ruido), size=N_os)
    
    # muy correlada
    # nn = sig.chirp(tt_os, 2*df, (N-1)*ts, fs/2)
    # nn = nn / np.sqrt(np.var(nn)) * np.sqrt(pot_ruido)
    
    # construimos la señal de entrada al ADC
    sr[:, rr]  = analog_sig + nn[:, rr] 
    # sr = analog_sig 
    
    # muestreo la señal analógica 1 cada OS muestras
    # sr = sr[::over_sampling]
    
    # cuantizo la señal muestreada
    srq[:, rr]  = q * np.round(sr[:, rr] /q)
    
    # ruido de cuantización
    nq[:, rr]  = srq[:, rr]  - sr[:, rr] 


#%% Presentación gráfica de los resultados
plt.close('all')

plt.figure(1)
plt.plot(tt_os, srq[:, 0] , lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)')
plt.plot(tt_os, sr[:, 0], linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ s_R = s + n $  (ADC in)')
plt.plot(tt_os, analog_sig, color='orange', ls='dotted', label='$ s $ (analog)')

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
bfrec_os = ff_os <= fs/2 * over_sampling
bfrec = ff <= fs/2



Nnq_mean = np.mean(np.mean(np.abs(ft_Nq)**2, axis=1))
nNn_mean = np.mean(np.abs(ft_Nn)**2)

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
plt.ylim((1.5*np.min(10* np.log10(2* np.array([Nnq_mean, nNn_mean]))),10))


plt.figure(3)
bins = 10
plt.hist(nq.flatten(), bins=bins)
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N_os*R/bins, N_os*R/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))


