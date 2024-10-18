#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:27:05 2024

@author: mariano
"""

import sounddevice as sd
from scipy.io.wavfile import write

# sd.query_devices()
dev_props = sd.query_devices(device = 3)

# Configuraciones
duration = 3  # Duración en segundos
fs = dev_props['default_samplerate']  # Frecuencia de muestreo
sd.default.device = (3,11)

# Grabar audio
print("Grabando...")
audio = sd.rec(int(duration * fs), samplerate = fs, channels = 1)
sd.wait()  # Esperar a que la grabación termine

# Guardar el audio en un archivo WAV
write("grabacion.wav", int(fs), audio)

print("Grabación finalizada y guardada en 'grabacion.wav'")
