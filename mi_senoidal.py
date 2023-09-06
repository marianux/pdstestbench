#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:55:33 2023

@author: mariano
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la simulación

fs = 250 # Hz
N = fs

delta_f = fs / N # Hz

Ts = 1/fs
T_simulacion = N * Ts # segundo


# Parámetros de la señal
fx = 10 # Hz
ax = 2 # V


# grilla temporal
tt = np.arange(start = 0, stop = T_simulacion, step = Ts)

xx = ax * np.sin( 2 * np.pi * fx * tt )

plt.plot( tt, xx, 'o--')

