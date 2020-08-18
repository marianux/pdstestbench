#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ...
@author: mariano
Descripción:
    
    Ejemplificaremos el uso de las herramientas que utilizaremos más
    frecuentemente en Procesamiento Digital de Señales (PDS).
    Estos ejemplos fueron pensadas para aquellos estudiantes que no
    hayan tenido contacto con Python.
    
"""

# Importación de módulos que utilizaremos en nuesto testbench:
# Una vez invocadas estas funciones, podremos utilizar los módulos a través del identificador que indicamos luego de "as". 
# Por ejemplo np.linspace() -> función linspace dentro e NumPy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdsmodulos as pds
import numbers


###################################
## Formas de incluir comentarios ##
###################################

# Comentarios con "#"

""" Bloques de comentario
    bla bla bla
    bla ...
"""

#%% Separación de bloques de código para ordenar tu trabajo "#%%"
# Definición de funciones: saltear al comienzo del ejemplo, y volver cuando 
# se invoquen.

def una_funcion_que_no_hace_nada(a):
    
    # hasta acá
    return(a)
 
def funcion_que_suma(aa, bb = 0, cc = 0):
    
    # hasta acá
    return(aa+bb+cc)
 
def funcion_que_concatena(s1, s2 = ''):
    
    # hasta acá
    return(s1+s2)
 

#%% Comienzo del ejemplo "#%%"

#####################################
# Uso de variables y tipos de datos #
#####################################

# flotantes, enteros o tipos numéricos
Fs = 1000.0 # Hz
N = 1000 # muestras

un_texto = 'señal 0'
otro_texto = "señal 1"

# concatenamos texto
mi_texto = un_texto + otro_texto

# una lista vacía
mi_lista = []

# la puedo llenar con datos "iterables" (googlea y averiguá más sobre esto ...)
mi_lista += un_texto
# pero posiblemente me interese más conservarlo como string, así que 
# fabrico una lista de listas
mi_lista += [otro_texto]
mi_lista += [Fs]

# Accedé a los elementos direccionando elementos de la lista

# el primero
mi_lista[0]
# el segundo
mi_lista[1]
# el ante-último
mi_lista[-1]
# el último
mi_lista[-2]

print(mi_lista)

# también podés seleccionar rangos de elementos:

# del primero al tercero
print(mi_lista[0:3])
#print(mi_lista[:3]) # otra forma

# los pares
print(mi_lista[0::2])
#print(mi_lista[::2]) # otra forma

# los impares
print(mi_lista[1::2])
    
# recorrerla al revés
print(mi_lista[::-1])
    
# podemos aprovechar que sea iterable con una sintaxis inline y muy 
# práctica de Python

[ print(aa) for aa in mi_lista ]
tipo = [ type(aa) for aa in mi_lista ]


# Tipos de datos del módulo Numpy: es un módulo para agregar los tipos 
# numéricos y métodos que se utilizan en Matlab para el álgebra matricial.
# Matrices, vectores, operaciones algebraicas, direccionamiento, etc.

# vector con una secuencia de enteros y flotantes
vector_tiempo_int = np.arange(10)
vector_tiempo = np.arange(10.0)

vector_tiempo_int = np.arange(5, 10)
vector_tiempo = np.arange(5.0, 10.0)

vector_tiempo = np.arange(5.0, 10.0, 0.1)

# armar vectores con secuencias numéricas nos resultará muy útil
# para simular por ejemplo un eje temporal o frecuencial

N = 1000
Fs = 1000
vector_tiempo = np.arange(0.0, N/Fs, 1/Fs)

# en este caso, el vector_tiempo podría ser los tiempos en los que un
# ADC muestrea sus canales analógicos.
# En consecuencia, tendremos una representación frecuencial desde 0 a 
# Fs/2, es decir la frecuencia de Nyquist.

vector_frecuencia = np.arange(0.0, Fs/2.0, N/Fs)

# ahora podemos simular que los canales están desconectados,
# o que una señal de ruido blanco, normalmente distribuido ingresa al ADC.
canales_ADC = 8
matriz_datos = np.random.normal(0,1.0,size = [N,canales_ADC])

# los vectores de booleanos son muy útiles para direccionar
bool_vector = abs(matriz_datos[:,0]) > 0.5

# como también son útiles los índices. NOTAR que nonzero devuelve una tupla.
# Una tupla es algo parecido a una lista (googlear para saber más). Luego de
# esa tupla invocamos al primer y único elemento. Como que desempaquetamos la 
# tupla. 
indices = np.nonzero(bool_vector)[0]

# luego podemos direccionar la matriz tanto con índices como con booleanos.
matriz_datos[bool_vector, :]
# matriz_datos[indices, :] # lo mismo

# más booleanos, creados de otras formas
es_numero = [ isinstance(aa, numbers.Number) for aa in mi_lista ]
es_texto = [ isinstance(aa, str) for aa in mi_lista ]

# llamado a funciones

# aa = 1, bb = 2 y cc ??
print(funcion_que_suma(1, 2))

# lo mismo, pero notar que podemos asegurar especificar a qué argumento
# nos referimos cuando invocamos a una función, como también que los argumentos
# que no reciben valores, toman los valores por defecto ( bb = 0).
print(funcion_que_suma(1, cc = 2))

##################################
# Algunas estructuras de control #
##################################

# otra forma de recorrer un tipo de dato iterable
for aa in mi_lista:
    
    print(aa)

# recorremos listas al estilo C mediante índices

for ii in range(len(mi_lista)):
    
    print(mi_lista[ii])
    print(mi_lista[len(mi_lista)-ii-1])


# lo mismo, pero más estilo Python se pueden recorrer tantos iterables 
# como quieras, solo tenés que empaquetarlos con *zip*

for aa,bb in zip(mi_lista, mi_lista[::-1]):
    
    print(aa)
    print(bb)

# luego de que te acostumbres un poco, esta última forma te parecerá
# más legible





