#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariano Llamedo Soria llamedom@frba.utn.edu.ar

En este script demostramos el uso de ventanas para el análisis espectral.
"""

import numpy as np
import scipy.signal.windows as win
import matplotlib.pyplot as plt
from scipy.fftpack import fft


N = 1000
Fs = 1000
tt = np.arange(0.0, N/Fs, 1/Fs)
ff = np.arange(0.0, Fs, N/Fs)

# ahora podemos simular que los canales están desconectados,
# o que una señal de ruido blanco, normalmente distribuido ingresa al ADC.
canales_ADC = 1
a0 = 1 # Volt
f0 = N/4 * Fs/N

dd = win.blackman(N)

bfrec = ff <= Fs/2

ft_dd = fft( dd )
# ft_dd = fft( dd, axis = 0 )

plt.close('all')

plt.figure(2)
plt.plot( ff[bfrec], np.abs(ft_dd[bfrec]) )
# plt.plot( ff[bfrec], 20*np.log10(np.abs(ft_dd[bfrec])) )
plt.ylabel('Módulo [¿Volts?]')
plt.xlabel('Frecuencia [Hz]')

plt.figure(1)
plt.plot( tt, dd )
plt.ylabel('#')
plt.xlabel('t [s]')

