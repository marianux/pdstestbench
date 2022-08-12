#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDS TS 8

En este script se analizará sesgo y varianza del Periodograma (Hayes 8.2.2)

@author: mariano
"""

#%% Configuración e inicio de la simulación

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from pandas import DataFrame
from IPython.display import HTML


# Datos generales de la simulación
fs = 1000.0 # frecuencia de muestreo (Hz)

# datos del ruido
pot_ruido = 2 # W

#%% Acá arranca la simulación

los_N = np.flip(np.array([10, 50, 100, 250, 500, 1000, 5000]), axis = 0)

cant_exp = 1000

tus_resultados_per = [ 
                   ['', ''], # <-- acá debería haber numeritos :)
                   ['', ''], # <-- acá debería haber numeritos :)
                   ['', ''], # <-- acá debería haber numeritos :)
                   ['', ''], # <-- acá debería haber numeritos :)
                   ['', ''], # <-- acá debería haber numeritos :)
                   ['', ''], # <-- acá debería haber numeritos :)
                   ['', ''], # <-- acá debería haber numeritos :)
                 ]


plt.close('all')

for ii in range(los_N.shape[0]):
    
    N = los_N[ii]
    # Generación de la señal a analizar ruido blanco Gaussiano (WGN)
    nn = np.random.normal(0, np.sqrt(pot_ruido), size=(N, cant_exp) )
    
    pN = 1/N*np.abs(np.fft.fft( nn, axis = 0 ))**2
    pNm = np.mean(pN, axis=1 )
    pNv = np.var(pN, axis=1 )

    pNmm = np.median(pNm)
    pNmv = np.median(pNv)
    tus_resultados_per[ii] = [pNmm, pNmv]

    df = fs/N # resolución espectral
    # grilla de sampleo frecuencial
    ff = np.linspace(0, (N-1)*df, N)

    plt.figure()

    bfrec = ff <= fs/2
    plt.plot( ff[bfrec], pNm[bfrec], ':x' )
    plt.plot( [ff[bfrec][0], ff[bfrec][-1]] , [pNmm, pNmm], '--', lw=2 )
    plt.title('N = {:d}; $s_P$ = {:3.3f}; $v_P$ = {:3.3f}'.format(N, 2-pNmm, pNmv))
    plt.ylim((1.8,2.2))


df = DataFrame(tus_resultados_per, columns=['$s_P$', '$v_P$'],
               index=los_N)

display(df)

