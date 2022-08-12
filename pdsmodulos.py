#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:52:18 2020

@author: Mariano Llamedo Soria llamedom@frba.utn.edu.ar

Descripción
-----------

En este módulo podrías incluir las funciones más generales que quieras usar desde todos los TP's
y notebooks que uses. Yo incluyo algunas que utilizaré en algunos ejemplos.

"""
from IPython.display import display, Math, Markdown

def pdsmodulos():
    
    print("hola mundo!")

def print_subtitle(strAux):
    
    display(Markdown('#### ' + strAux))

def print_markdown(strAux):
    
    display(Markdown(strAux))

def print_latex(strAux):
    
    display(Math(strAux))

    


# Esto puede utilizarse dentro de los archivos de módulo para hacer 
# pruebas de funcionalidad de las funciones o clases que se incluyen
# dentro de un módulo
if __name__ == "__main__":
    
    print_subtitle('Subtituilo1')
    print_markdown('Con este texto...')
    print_latex('Subtituilo1')
    
    