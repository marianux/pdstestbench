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
from pytc2.sistemas_lineales import plot_plantilla


# Datos generales de la simulación
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras
R = 1 # realizaciones o experimentos

# filtro analógico
filter_size = 300

# cantidad de veces más densa que se supone la grilla temporal para tiempo "continuo"
over_sampling = 4
N_os = N*over_sampling

# Datos del ADC
B = 6 # bits
Vf = 2 # rango simétrico de +/- Vf Volts
q = Vf/2**(B-1) # paso de cuantización de q Volts

# datos del ruido
kn = 20
pot_ruido = q**2/12 * kn # Watts (potencia de la señal 1 W)

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

#%% Simulación del filtro analógico

# Tipo de aproximación.
        
aprox_name = 'butter'
# aprox_name = 'cheby1'
# aprox_name = 'cheby2'
# aprox_name = 'ellip'

# Requerimientos de plantilla
filter_type = 'lowpass'

ftran = 0.1
fstop = np.min([1/over_sampling + ftran/2, 1/over_sampling * 5/4])  #

fpass = np.max([fstop - ftran/2, fstop * 3/4]) # 
ripple = 0.5 # dB
attenuation = 40 # dB

# como usaremos filtrado bidireccional, alteramos las restricciones para
# ambas pasadas
ripple = ripple / 2 # dB
attenuation = attenuation / 2 # dB


if aprox_name == 'butter':

    # order, wcutof = sig.buttord( 2*np.pi*fpass*fs/2, 2*np.pi*fstop*fs/2, ripple, attenuation, analog=True)
    orderz, wcutofz = sig.buttord( fpass, fstop, ripple, attenuation, analog=False)

elif aprox_name == 'cheby1':

    # order, wcutof = sig.cheb1ord( 2*np.pi*fpass*fs/2, 2*np.pi*fstop*fs/2, ripple, attenuation, analog=True)
    orderz, wcutofz = sig.cheb1ord( fpass, fstop, ripple, attenuation, analog=False)
    
elif aprox_name == 'cheby2':

    # order, wcutof = sig.cheb2ord( 2*np.pi*fpass*fs/2, 2*np.pi*fstop*fs/2, ripple, attenuation, analog=True)
    orderz, wcutofz = sig.cheb2ord( fpass, fstop, ripple, attenuation, analog=False)
    
elif aprox_name == 'ellip':
   
    # order, wcutof = sig.ellipord( 2*np.pi*fpass*fs/2, 2*np.pi*fstop*fs/2, ripple, attenuation, analog=True)
    orderz, wcutofz = sig.ellipord( fpass, fstop, ripple, attenuation, analog=False)


# Diseño del filtro digital

filter_sos = sig.iirfilter(orderz, wcutofz, rp=ripple, rs=attenuation, 
                            btype=filter_type, 
                            analog=False, 
                            ftype=aprox_name,
                            output='sos')


# sin filtro
# filter_sos = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]

#%% Acá arranca la simulación

# grilla de sampleo temporal
tt = np.linspace(0, (N-1)*ts, N)
tt_os = np.linspace(0, (N-1)*ts, N_os)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)
ff_os = np.linspace(0, (N_os-1)*df, N_os)

# Concatenación de matrices:
# guardaremos las señales creadas al ir poblando la siguiente matriz vacía

analog_sig = np.sin( 2*np.pi*10*df*tt_os )

# normalizo en potencia
analog_sig = analog_sig / np.sqrt(np.var(analog_sig))

nq = np.zeros((N_os,R))
nn = np.zeros((N_os,R))
sr = np.zeros((N_os,R))
srq = np.zeros((N_os,R))

nqf = np.zeros((N_os,R))
nqfd = np.zeros((N,R))
srf = np.zeros((N_os,R))
srfd = np.zeros((N,R))
srqf = np.zeros((N_os,R))
srqfd = np.zeros((N,R))

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

    # limito en la banda de muestreo objetivo
    srf[:, rr]  = sig.sosfiltfilt(filter_sos, sr[:, rr])

    # muestreo la señal analógica 1 cada OS muestras
    srfd[:, rr]  = srf[:, rr][::over_sampling]
    
    # cuantizo la señal muestreada
    srq[:, rr]  = q * np.round(sr[:, rr] /q)

    # limito en la banda de muestreo objetivo
    srqf[:, rr]  = sig.sosfiltfilt(filter_sos, srq[:, rr])

    # muestreo la señal analógica 1 cada OS muestras
    srqfd[:, rr]  = srqf[:, rr][::over_sampling]
    
    # ruido de cuantización
    nq[:, rr]  = srq[:, rr]  - sr[:, rr] 
    nqf[:, rr]  = sig.sosfiltfilt(filter_sos, nq[:, rr])
    nqfd[:, rr]  = nqf[:, rr][::over_sampling]
    # nqfd[:, rr]  = srqfd[:, rr]  - srfd[:, rr] 


#%% Presentación gráfica de los resultados
plt.close('all')

plt.figure(1)
plt.plot(tt_os, analog_sig, color='orange', ls='dotted', label='$ s $ (analog)')
plt.plot(tt_os, sr[:, 0], linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ s_R = s + n $')
plt.plot(tt, srqfd[:, 0] , lw=2, label='$ s_{Qfd} = Q_{B,V_F}\{s_R\} $')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

# %% figure(2)

plt.figure(2)
ft_SR = 1/N_os*np.fft.fft( sr, axis = 0 )
ft_Srq = 1/N_os*np.fft.fft( srq, axis = 0 )
ft_As = 1/N_os*np.fft.fft( analog_sig, axis = 0)
ft_Nq = 1/N_os*np.fft.fft( nq, axis = 0 )
ft_Nn = 1/N_os*np.fft.fft( nn, axis = 0 )
ft_Srqf = 1/N_os*np.fft.fft( srqf, axis = 0 )
ft_Nqf = 1/N_os*np.fft.fft( nqf, axis = 0 )
bfrec_os = ff_os <= fs/2 * over_sampling

ft_Srqfd = 1/N*np.fft.fft( srqfd, axis = 0 )
ft_srfd = 1/N*np.fft.fft( srfd, axis = 0 )
ft_Nqfd = 1/N*np.fft.fft( nqfd, axis = 0 )
bfrec = ff <= fs/2

# Nnq_mean = np.mean(np.abs(ft_Nq)**2)
Nnq_mean = np.mean(np.mean(np.abs(ft_Nq)**2, axis=1))
nNn_mean = np.mean(np.mean(np.abs(ft_Nn)**2, axis=1))
# nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.abs(ft_As[bfrec_os])**2), color='orange', ls='dotted', label='$ s $ (analog)' )
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Nn)**2, axis=1)[bfrec_os]), ':r')
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_SR)**2, axis=1)[bfrec_os]), ':g', label='$ s_R = s + n $  (ADC in)' )

plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Srq)**2, axis=1)[bfrec_os]), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)' )

plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Nq)**2, axis=1)[bfrec_os]), ':c')

plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--', color = 'orange', lw = 2, label='$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--', color = 'red', lw = 2, label= '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)) )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()
# suponiendo valores negativos de potencia ruido en dB
plt.ylim((1.5*np.min(10* np.log10(2* np.array([Nnq_mean, nNn_mean]))),10))

# %% figure(3)

plt.figure(3)
bins = 10
plt.hist(nq.flatten(), bins=bins)
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N_os*R/bins, N_os*R/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))


#%% figure(4)

plt.figure(4)

plt.plot( ff[bfrec], 10* np.log10(2*np.mean(np.abs(ft_srfd)**2, axis=1)[bfrec]), ':', label='$ s_Rfd = filt(s_R) $')
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Srq)**2, axis=1)[bfrec_os]), ':', label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Srqf)**2, axis=1)[bfrec_os]), lw=2, label='$ s_Q $ (filt.)' )


# plt.plot( np.array([ ff_os[bfrec_os][0], ff_os[bfrec_os][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Nq)**2, axis=1)[bfrec_os]), ':c', label='$ n_Q $')
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Nqf)**2, axis=1)[bfrec_os]), ':g', label='$ n_{Qf} $')
plt.plot( ff[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Nqfd)**2, axis=1)[bfrec]), ':r', label='$ n_{Qfd} $')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

# plt.plot(ff_os[bfrec_os], 20 * np.log10(abs(hh)), '--k', label='dec. filter')

plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--', color = 'orange', lw = 2, label='$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--', color = 'red', lw = 2, label= '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)) )


plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()
# suponiendo valores negativos de potencia ruido en dB
# plt.ylim((1.5*np.min(10* np.log10(2* np.array([Nnq_mean, nNn_mean]))),10))

ylim_aux = plt.ylim()

ax = plt.gca()

yrange = ylim_aux[1] - ylim_aux[0]
# flechas
ax.annotate('',
            xy=(fs/2, ylim_aux[0] + 0.1*yrange), xycoords='data',
            xytext = (fs/2*over_sampling, ylim_aux[0] + 0.1*yrange), textcoords='data',
            arrowprops={'arrowstyle': '<|-|>', 'lw' : 0.5})

# texto
ax.annotate('oversampling',
            xy=(fs/4*(over_sampling-1) , ylim_aux[0] + 0.15*yrange),
            horizontalalignment = 'center', 
            verticalalignment =   'center')


# %% figure(6)

# Análisis del filtro
wrad, hh = sig.sosfreqz(filter_sos, worN=(N_os//2)+1)
ww = wrad / np.pi

plt.figure(6)

plt.plot(ww, 20 * np.log10(abs(hh)))

plot_plantilla(filter_type = 'lowpass' , fpass = fpass, ripple = ripple/2, fstop = fstop, attenuation = attenuation/2, fs = fs)

plt.title('FIR designed by window method')
plt.xlabel('Frequencia normalizada')
plt.ylabel('Modulo [dB]')
plt.grid(which='both', axis='both')

axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()


