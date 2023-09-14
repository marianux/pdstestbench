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
from pytc2.sistemas_lineales import plot_plantilla


# Datos generales de la simulación
fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras
R = 10 # realizaciones o experimentos

# filtro de diezmado
filter_size = 300

# cantidad de veces más densa que se supone la grilla temporal para tiempo "continuo"
over_sampling = 32
# extra data para suplir los transitorios del filtro de diezmado.
N_osp = N*over_sampling + 2*filter_size
N_os = N*over_sampling 

# Datos del ADC
B = 6 # bits
Vf = 2 # rango simétrico de +/- Vf Volts
q = Vf/2**(B-1) # paso de cuantización de q Volts

# datos del ruido
kn = 1/10
pot_ruido = q**2/12 * kn # Watts (potencia de la señal 1 W)

ts = 1/fs # tiempo de muestreo
df = fs/N # resolución espectral

#%% Acá arranca la simulación

# grilla de sampleo temporal
tt = np.linspace(0, (N-1)*ts, N)
# tt_os = np.linspace(0, (N-1)*ts, N_osp)
tt_os = np.linspace(-filter_size*ts/over_sampling, (N-1)*ts+filter_size*ts/over_sampling, N_osp)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)
ff_os = np.linspace(0, (N_os-1)*df, N_os)

# Concatenación de matrices:
# guardaremos las señales creadas al ir poblando la siguiente matriz vacía

fa = 10
analog_sig = np.sin( 2*np.pi*fa*df*tt_os )

# normalizo en potencia
analog_sig = analog_sig / np.sqrt(np.var(analog_sig))

# nn2 = np.sin( 2*np.pi*(200)*df*tt_os )

# nn2 = nn2 / np.sqrt(np.var(nn2))

#%% Diseño del filtro de diezmado

# FIR Tipo 2 fuerzo filter_size par
antisymmetric = False

ftran = 0.1
fstop = np.min([1/over_sampling + ftran/2, 1/over_sampling * 5/4])  #

fpass = np.max([fstop - ftran/2, fstop * 3/4]) # 
ripple = 0.5 # dB
attenuation = 40 # dB

# pasa bajo
frecs = [0.0,  fpass,     fstop,           1.0]
# la mitad de la att porque usaremos filtfilt
gains = [0,   -ripple/2, -attenuation/2,  -attenuation/2] # dB

gains = 10**(np.array(gains)/20)



# FIR design

#  método simple: ventanas
# algunas ventanas para evaluar
#win_name = 'boxcar'
#win_name = 'hamming'
win_name = 'blackmanharris'
#win_name = 'flattop'
# 
# restricción de filtro tipo II
gains[-1] = 0
num = sig.firwin2(filter_size, frecs, gains, window=win_name, antisymmetric=antisymmetric  )

# cuadrados mínimos (orden impar)
# num = sig.firls((filter_size//2)*2+1, frecs, gains, fs=2)

# equirriple: Remez exchange
# num = sig.remez(filter_size, frecs, [1, 0], weight=[5,1], fs=2)

# sin filtro
# num = [1.0, 0.0]
den = 1.0

#%% arranca la experimentación

nq = np.zeros((N_osp,R))
nn = np.zeros((N_osp,R))
sr = np.zeros((N_osp,R))
srq = np.zeros((N_osp,R))

srfd = np.zeros((N,R))
srqf = np.zeros((N_osp,R))
nqf = np.zeros((N_osp,R))
srqfd = np.zeros((N,R))
nqfd = np.zeros((N,R))

for rr in range(R):
    
    # Generación de la señal de interferencia
    # incorrelada
    nn[:, rr] = np.random.normal(0, np.sqrt(pot_ruido), size=N_osp)
    # nqi = np.random.uniform(low=-q/2, high=q/2, size=N_os)
    
    # muy correlada
    # nn = sig.chirp(tt_os, 2*df, (N-1)*ts, fs/2)
    # nn = nn / np.sqrt(np.var(nn)) * np.sqrt(pot_ruido)
    
    # construimos la señal de entrada al ADC
    sr[:, rr] = analog_sig + nn[:, rr]
    # sr = analog_sig 
    
    #%% Sigma-Delta ADC
    
    wh = np.zeros(N_osp)
    vo = np.zeros_like(wh)
    vi = np.zeros_like(wh)
    # pérdidas del integrador
    k = 1
    
    for ii in range(1, N_osp):
        
        # diferencia
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

    srqf[:, rr] = sig.filtfilt(num, 1, wh)
    srqfd[:, rr] = srqf[:, rr][filter_size:-filter_size:over_sampling]
    
    # 
    # ruido de cuantización
    # nq[:, rr] =  srq - vo[::over_sampling]
    # nq =  nq[::over_sampling]
    nq[:, rr] =  np.roll(srq[:, rr],-1) - sr[:, rr]
    

    nqf[:, rr] = sig.filtfilt(num, 1, nq[:, rr])
    nqfd[:, rr] = nqf[:, rr][filter_size:-filter_size:over_sampling]
    srfd[:, rr] = srqfd[:, rr] - nqfd[:, rr]


# filtro los extremos anti-transitorios
nq = nq[filter_size:-filter_size,:]
nn = nn[filter_size:-filter_size,:]
sr = sr[filter_size:-filter_size,:]
srq = srq[filter_size:-filter_size,:]
srqf = srqf[filter_size:-filter_size,:]
nqf = nqf[filter_size:-filter_size,:]

tt_os = tt_os[filter_size:-filter_size]
analog_sig = analog_sig[filter_size:-filter_size]
wh = wh[filter_size:-filter_size]

#%% Presentación gráfica de los resultados
plt.close('all')

plt.figure(1)
plt.plot(tt, srqfd[:,0], lw=2, linestyle='', color='blue', marker='o', markersize=5, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='ADC out (diezmada)')
plt.plot(tt_os, wh, lw=1, linestyle='--', color='red', marker='o', markersize=3, markerfacecolor='none', markeredgecolor='red', fillstyle='none', label='$ \hat{w} $')
plt.plot(tt, srfd[:,0], lw=1, color='black', ls='dotted', label='$ s $ (analog)')
# plt.plot(tt_os, vo, lw=1, color='green', ls='--', marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ v_o $ (int. out)')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

#%% figure(2)

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

plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.abs(ft_As[bfrec_os])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
plt.plot( np.array([ ff_os[bfrec_os][0], ff_os[bfrec_os][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_SR)**2, axis=1)[bfrec_os]), ':g', label='$ s_R = s + n $' )
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Srq)**2, axis=1)[bfrec_os]), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
plt.plot( np.array([ ff_os[bfrec_os][0], ff_os[bfrec_os][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Nn)**2, axis=1)[bfrec_os]), ':r')
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Nq)**2, axis=1)[bfrec_os]), ':c')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

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

#%% figure(3)

plt.figure(3)

plt.plot( ff[bfrec], 10* np.log10(2*np.mean(np.abs(ft_srfd)**2, axis=1)[bfrec]), ':', label='$ s_Rfd = filt(s_R) $')
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Srq)**2, axis=1)[bfrec_os]), ':', label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Srqf)**2, axis=1)[bfrec_os]), lw=2, label='$ s_Q $ (filt.)' )


# plt.plot( np.array([ ff_os[bfrec_os][0], ff_os[bfrec_os][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Nq)**2, axis=1)[bfrec_os]), ':c', label='$ n_Q $')
plt.plot( ff_os[bfrec_os], 10* np.log10(2*np.mean(np.abs(ft_Nqf)**2, axis=1)[bfrec_os]), ':g', label='$ n_{Qf} $')
plt.plot( ff[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Nqfd)**2, axis=1)[bfrec]), ':r', label='$ n_{Qfd} $')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

# plt.plot(ff_os[bfrec_os], 20 * np.log10(abs(hh)), '--k', label='dec. filter')

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

#%% figure(4)

plt.figure(4)

bfrec_os2 = ff_os <= fs/2

Nnqfd_mean = np.mean(np.mean(np.abs(ft_Nqfd)**2, axis=1))

plt.plot( ff_os[bfrec_os2], 10* np.log10(2*np.abs(ft_As[bfrec_os2])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
plt.plot( ff_os[bfrec_os2], 10* np.log10(2*np.mean(np.abs(ft_Nn)**2, axis=1)[bfrec_os2]), ':r', label='$ n $ (ruido)')
plt.plot( ff[bfrec], 10* np.log10(2*np.mean(np.abs(ft_srfd)**2, axis=1)[bfrec]), ':g', label='$ s_{Rfd} = filt(s + n) $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Srqfd)**2, axis=1)[bfrec]), lw=2, label='$ s_{Qfd} = Q_{B,V_F}\{s_{Rf}\} $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.mean(np.abs(ft_Nqfd)**2, axis=1)[bfrec]), ':c', label='$ n_{Qfd} = S_{Qf} - s_{Rf}$')

plt.plot( np.array([ ff_os[bfrec_os2][0], ff_os[bfrec_os2][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--k', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( np.array([ ff_os[bfrec_os2][0], ff_os[bfrec_os2][-1] ]), 10* np.log10(2* np.array([Nnqfd_mean, Nnqfd_mean]) ), '--c', label='$ \overline{n_Qfd} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnqfd_mean)) )

# 0 dB ref
plt.plot( np.array([ ff_os[bfrec_os2][0], ff_os[bfrec_os2][-1] ]), (0, 0), ':k' )
ax = plt.gca()

SNR_target_db = -10* np.log10(pot_ruido) # 0 dB referencia de señal
N_db = 10* np.log10(N)
Os_db = 10* np.log10(over_sampling)
SNR_visual_db = SNR_target_db + Os_db + N_db
xgap = np.round(0.05*fs/2)
xann = fa + xgap

Nnq_mean_db = 10* np.log10(2*Nnq_mean)
Nnqfd_mean_db = 10* np.log10(2*Nnqfd_mean)

# flechas
ax.annotate('',
            xy=(xann, 0), xycoords='data',
            xytext = (xann, -SNR_visual_db), textcoords='data',
            arrowprops={'arrowstyle': '<|-|>', 'lw' : 0.5})

ax.annotate('',
            xy=(xann + xgap/3 , 0), xycoords='data',
            xytext = (xann + xgap/3, -SNR_target_db), textcoords='data',
            arrowprops={'arrowstyle': '<|-|>', 'lw' : 0.5})

ax.annotate('',
            xy=(xann + xgap/3 , Nnq_mean_db), xycoords='data',
            xytext = (xann + xgap/3, Nnqfd_mean_db), textcoords='data',
            arrowprops={'arrowstyle': '<|-|>', 'lw' : 0.5})

# texto
ax.annotate('SNR = 10log($N.O_S.q^2/12$) = {:3.3g} + {:3.3g} + {:3.3g} = {:3.3g} (dB) '.format(N_db, Os_db, SNR_target_db, SNR_visual_db),
            xy=(xann + xgap/3 , -SNR_visual_db*3/5),
            horizontalalignment = 'left', 
            verticalalignment =   'center')

ax.annotate('SNR = 10log($q^2/12$) = {:3.3g} (dB) '.format(SNR_target_db),
            xy=(xann + xgap*2/3 , -SNR_target_db/2),
            horizontalalignment = 'left', 
            verticalalignment =   'center')

ax.annotate('SNR_G = {:3.3g} (dB) '.format(Nnq_mean_db - Nnqfd_mean_db),
            xy=(xann + xgap*2/3 , Nnq_mean_db-(Nnq_mean_db - Nnqfd_mean_db)/2),
            horizontalalignment = 'left', 
            verticalalignment =   'center')

plt.ylim(ylim_aux)

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')

axes_hdl = plt.gca()
axes_hdl.legend()

#%% figure(5)

plt.figure(5)
bins = 10
# plt.hist(nq.flatten(), bins=2*bins)
plt.hist(nqf.flatten()/(q/2), bins=2*bins)
# plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N_os*R/bins, N_os*R/bins, 0]), '--r' )
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N*R/bins, N*R/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')



# %% figure(6)

# Análisis del filtro
wrad, hh = sig.freqz(num, den, worN=(N_os//2)+1)
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


