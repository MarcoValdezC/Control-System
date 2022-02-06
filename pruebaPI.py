# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 21:07:50 2022

@author: marco
"""
import os
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def inverted_pendulum(r):
    
    '''Time parameters'''
    dt = 0.005  # Tiempo de muestreo (5ms)
    ti = 0.0  # Tiempo inicial de la simulación (0s)
    tf = 10.0  # Tiempo inicial de la simulación (10s)
    n = int((tf - ti) / dt) + 1  # Número de muestras
    t = np.linspace(ti, tf, n)  # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)
    
    '''Dynamic parameters'''
    m = 0.5  # Masa del pendulo (kg)
    M = 0.7  # Masa del carro (kg) 
    l = 1.0  # Longitud de la barra del péndulo (m)
    lc = 0.3  # Longitud al centro de masa del péndulo (m)
    b1 = 0.05  # Coeficiente de fricción viscosa pendulo
    b2 = 0.06  # Coeficiente de friccion del carro
    g = 9.81  # Aceleración de la gravedad en la Tierra
    I = 0.006  # Tensor de inercia del péndulo

    '''State variables'''
    z = np.zeros((n, 4))

    '''Control vector'''
    u = np.zeros((2, n))
    
    '''Initial conditions'''
    z[0, 0] = 0  # Posición inicial del carro (m)
    z[0, 1] = 0 # Posición inicial del péndulo (rad)
    z[0, 2] = 0  # Velocidad inicial del carro (m/s) 
    z[0, 3] = 0  # Velocidad angular del péndulo (rad/s)
    ie_th = 0 #Inicialización de error integral de posición de carro
    ie_x = 0#Inicialización de error integral de posición de péndulo
    
    ise=0
    ise_next=0
    iadu=0
    iadu_next=0

    '''State equation'''
    zdot = [0, 0, 0, 0]
    
    '''Dynamic simulation'''
    for i in range(n - 1):
        '''Current states'''
        x = z[i, 0]  # Posicion del carro
        th = z[i, 1]   # Posición del péndulo
        x_dot= z[i, 2] # Velocidad del carro
        th_dot = z[i, 3]  # Velocidad del péndulo

        '''Controller'''
        e_x = 0 - x #Error de posición de carro
        e_x_dot = 0 - x_dot #Error de velocidad de carro
        e_th = np.pi/2-th #Error de posicón angular
        e_th_dot = 0 - th_dot #Error de velocidad angular 

        '''Ganancias del controlador del carro'''
        Kp = r[0]
        Kd = r[1]
        Ki = r[2]

        '''Ganancias del controlador del péndulo'''
        Kp1 =r[3]
        Kd1 =r[4]
        Ki1 =r[5]

        u[0, i] = Kp * e_x + Kd * e_x_dot + Ki * ie_x #Señal de control del actuador del carro
        u[1, i] = Kp1 * e_th + Kd1 * e_th_dot + Ki1 * ie_th #Señal de control del actuador del péndulo
        
        print(u[0,i])
        print(u[1,i])

        MI = np.array([[M + m,-m * lc * np.sin(th)], [-m * lc * np.sin(th), I + m * lc ** 2]])  # Matriz de inercia
        MC = np.array([[b1, -m * lc * np.cos(th) * th_dot], [0, b2]])  # Matriz de Coriollis
        MG = np.array([[0], [m * g * lc * np.cos(th)]])  # Vector de gravedad

        array_dots = np.array([[x_dot], [th_dot]])#Vector de velocidades
        MC2 = np.dot(MC, array_dots)

        ua = np.array([[u[0, i]], [u[1, i]]])
        aux1 = ua - MC2 - MG
        Minv = inv(MI)
        aux2 = np.dot(Minv, aux1) #Varables de segundo grado /(doble derivada)

        '''System dynamics'''
        zdot[0] = x_dot #Velocidad del carro
        zdot[1] = th_dot #Velocidad del péndulo
        zdot[2] = aux2[0, :] #Aceleración del carro
        zdot[3] = aux2[1, :] #Aceleración del péndulo

        '''Integrate dynamics'''
        z[i + 1, 0] = z[i, 0] + zdot[0] * dt
        z[i + 1, 1] = z[i, 1] + zdot[1] * dt
        z[i + 1, 2] = z[i, 2] + zdot[2] * dt
        z[i + 1, 3] = z[i, 3] + zdot[3] * dt
        ie_th = ie_th + e_th * dt
        ie_x = ie_x+e_x*dt

        ise=ise_next+(e_th**2)*dt+(e_x**2)*dt
        iadu=iadu_next+ (abs(u[0,i]-u[0,i-1]))*dt+(abs(u[1,i]-u[1,i-1]))*dt
        g=0
        if(ise>=20):
            ie=20
            g+=1
        else:
            ie=ise
            g+=0
        if(iadu>=0.8):
            ia=0.8
            g+=1
        else:
            ia=iadu
            g+=0
   
        ise_next=ie
        iadu_next=ia
    u[:,n - 1] = u[:,n - 2] #Actualizar señal de control
    
    #print(z[:, 0])
    return np.array([ise_next, iadu_next]),g

p=np.array([1,5,3,10,5,6])

h=inverted_pendulum(p)
print(h)