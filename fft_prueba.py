#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:52:18 2020

@author: Mariano Llamedo Soria llamedom@frba.utn.edu.ar

En este script demostramos el uso de la FFT, es decir de la implementación
computacionalmente eficiente de la transformada discreta de Fourier, o DFT.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from pdsmodulos import print_markdown, print_subtitle


N = 1000
Fs = 1000
tt = np.arange(0.0, N/Fs, 1/Fs)
ff = np.arange(0.0, Fs, N/Fs)

# ahora podemos simular que los canales están desconectados,
# o que una señal de ruido blanco, normalmente distribuido ingresa al ADC.
canales_ADC = 1
a0 = 1 # Volt
f0 = N/4 * Fs/N

# dd = np.sin(2*np.pi*f0*tt)
# dd = np.random.uniform(-np.sqrt(12)/2, +np.sqrt(12)/2, size = [N,canales_ADC])
# dd = np.random.normal(0, 1.0, size = [N,canales_ADC])

DD = fft( dd )

print_subtitle('Teorema de Parseval')

plt.close('all')

plt.figure(1)
plt.plot( ff, np.abs(DD) )
plt.ylabel('Módulo [¿Volts?]')
plt.xlabel('Frecuencia [Hz]')

plt.figure(2)
plt.plot( ff, np.abs(DD)**2 )
plt.ylabel('Densidad de Potencia [¿W/Hz?]')
plt.xlabel('Frecuencia [Hz]')


