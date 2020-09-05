#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:44:40 2020

@author: Mariano Llamedo Soria llamedom@frba.utn.edu.ar

Analizamos el módulo de una ventana rectangular de acuerdo al resultado 
calculado algebraicamente.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

N = 1000
kk = np.arange(start=-10, stop=10, step= 1/10 )

kernel_dirchlet = np.sin(np.pi * kk) / np.sin(np.pi * kk / N)

# kernel_dirchlet[ kk == 0 ] = N

plt.close('all')

plt.figure(1)
plt.plot(kk, np.abs(kernel_dirchlet))
plt.plot( np.array([-10, 10]), np.array([0, 0]), ':k' )
# plt.plot( np.array([0, N]), np.array([q/2, q/2]), '--r' )
# plt.title( 'Ruido de cuantización para {:d} bits - V_R={:3.1f} V - q = {:3.3f} V'.format(B, Vref, q))
# plt.ylabel('Amplitud ruido [V]')
# plt.xlabel('Muestras [#]')
