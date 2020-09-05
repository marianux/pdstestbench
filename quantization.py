#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:44:40 2020

@author: Mariano Llamedo Soria llamedom@frba.utn.edu.ar

Simulamos el comportamiento del ruido de cuantización que podemos encontrar
en un conversor analógico digital (ADC) de B bits. Comparamos las estimaciones
de sus estadísticos con las que deberían tener según la distribución de 
sus valores (uniforme). Visualizamos algunos resultados, como también
el histograma de la señal de ruido.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

N = 1000; # muestras
B = 8 # bits
Vref = 3.3 # Volts

q = Vref/2**B

nn = np.random.uniform(-q/2,high=q/2, size=N)

nn_m = np.mean(nn)
nn_var = np.var(nn)

print('Media de la distribución: 0 -- Estimación de la media: {:g}'.format(nn_m) )
print('Varianza de la distribución (q^2/12): {:g} -- Estimación de la varianza: {:g}'.format(q**2/12, nn_var) )

nn_ac = sig.correlate( nn, nn)

plt.close('all')

plt.figure(1)
plt.plot(nn)
plt.plot( np.array([0, N]), np.array([-q/2, -q/2]), '--r' )
plt.plot( np.array([0, N]), np.array([q/2, q/2]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - V_R={:3.1f} V - q = {:3.3f} V'.format(B, Vref, q))
plt.ylabel('Amplitud ruido [V]')
plt.xlabel('Muestras [#]')

plt.figure(2)
bins = 10
plt.hist(nn, bins=bins)
plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - V_R={:3.1f} V - q = {:3.3f} V'.format(B, Vref, q))

plt.figure(3)
plt.plot(nn_ac)
plt.title( 'Su secuencia de autocorrelación'.format(B,Vref))
plt.ylabel('Autocorrelacion [#]')
plt.xlabel('Demora [#]')



