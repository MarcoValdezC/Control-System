# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:57:24 2022

@author: marco
"""
import os
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.integrate import odeint 

import time
import math
import pylab as py


from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib import pyplot as plt

import random
from drawnow import *

def double_pendulum(h):
    print(h)
    '''Time parameters''' #Parametros temporales
    dt = 0.005  # Tiempo de muestreo (5ms)
    ti = 0.0  # Tiempo inicial de la simulación (0s)
    tf =10  # Tiempo final de la simulación (12.25s)
    n = int((tf - ti) / dt) + 1  # Número de muestras
    t = np.linspace(ti, tf, n)  # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)
    
    '''Dynamic parameters''' #Parametros dinamicos
    m1 = 0.5  # Masa de la barra 1(kg)
    m2= 0.5 #Masa de la barra 2 (kg)
    l1 = 1.0  # Longitud de la barra 1 (m)
    lc1 = 0.5  # Longitud al centro de masa de la barra 2 (m)
    l2= 1.0 #.0Longitud de la baraa 2 (m)
    lc2=0.3 #Longitud al centro de masa de la barra 2(m)
    b1 = 0.05  # Coeficiente de fricción viscosa de la barra 1
    b2= 0.02 #Coeficiente de fricción viscosa de la barra 2
    gravi = 9.81  # Aceleración de la gravedad en la Tierra
    I1 = 0.006  # Tensor de inercia del péndulo 1
    I2= 0.004 #Tensor de inercia del péndulo 2

    ''' Cinematica inversa'''

    r=0.2
    #ro=r*np.cos(3*t)

    '''Ecuaciones paramétricas de circunferencia'''
    Xp =1.4+ r*np.cos(t)
    Yp =0.2+ r*np.sin(t)
    '''Ecuaciones paramétricas de rosa de 3 petalos
    Xp =1.4+ ro*np.cos(t)
    Yp =0.2+ ro*np.sin(t)   '''

    #Ecuaciones pametricas lemniscata


    '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Calculo del Modelo Cinematico Inverso de Posicion
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

    #Variable Articular 2
    cosq2=(Xp**2+Yp**2-l1**2-l2**2)/(2*l1*l2)
    teta_rad_inv2=np.arctan2((1-cosq2**2)**(1/2),cosq2)

    teta_rad_inv1= np.arctan2(Xp,-Yp)-np.arctan2(l2*np.sin(teta_rad_inv2),(l1+l2*np.cos(teta_rad_inv2)));

    #teta_rad_inv2 =np.arccos((Xp**2+Yp**2-(l1**2+l2**2))/2*l1*l2)
    teta_grad_inv2=teta_rad_inv2*180/np.pi


    #Variable Articular 1 
    '''alfa=np.arctan2(Xp,Yp)
    beta=np.arccos((np.multiply(l1,l1)+np.multiply(l2,l2)-(np.multiply(Xp,Xp)+np.multiply(Yp,Yp)))/(2*l1*l2))
    gamma=np.arcsin((l2*np.sin(beta))/np.sqrt(np.multiply(Xp,Xp)+np.multiply(Yp,Yp)))
    '''

    #teta_rad_inv1=np.arctan2(-Yp,Xp)-np.arctan2(l2*np.sin(teta_rad_inv2),l1+l2*np.cos(teta_rad_inv2))
    teta_grad_inv1=teta_rad_inv1*180/np.pi
    '''Cinematica Diferencial inversa'''
    #Rosa de 3 petalos 
    #dx=-r*(3*np.sin(3*t)*np.cos(t)+np.cos(3*t)*np.sin(t))
    #dy=-r*(3*np.sin(3*t)*np.sin(t)-np.cos(3*t)*np.cos(t))
    #Circunferencia 
    dx=-r*np.sin(t)
    dy=r*np.cos(t)

    t1_dot=((np.sin(teta_rad_inv1+teta_rad_inv2))/(l1*np.sin(teta_rad_inv2))*dx)-((np.cos(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2))*dy)
    t2_dot=-(((l1*np.sin(teta_rad_inv1)+l2*np.sin(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2))*dx)+(((l1*np.cos(teta_rad_inv1)+l2*np.cos(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2))*dy)
    '''Cinematica Aceleración Inversa'''
    #Aceleración circunferencia
    ddx=-r*np.cos(t)
    ddy=-r*np.sin(t)

    #Aceleración rosa de 3 petalos
    #ddx=-r*(10*np.cos(3*t)*np.cos(t)-6*np.sin(3*t)*np.sin(t))
    #ddy=-r*(10*np.cos(3*t)*np.sin(t)+6*np.sin(3*t)*np.cos(t))

    #Jacobiano inverso 
    #Jinv= [[(np.sin(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2)), (-np.cos(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2))],[-(l1*np.sin(teta_rad_inv1)+l2*np.sin(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2),(l1*np.cos(teta_rad_inv1)+l2*np.cos(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2)]]
    #Jt=np.array([[-r*np.cos(t)-(t1_dot**2*(-l1*np.sin(teta_rad_inv1)-l2*np.sin(teta_rad_inv1+teta_rad_inv2))-2*t1_dot*t2_dot*l2*np.sin(teta_rad_inv1+teta_rad_inv2)-t2_dot**2*(l2*np.sin(teta_rad_inv1+teta_rad_inv2)))],[-r*np.sin(t)-(t1_dot**2*(l1*np.cos(teta_rad_inv1)-l2*np.cos(teta_rad_inv1+teta_rad_inv2))-2*t1_dot*t2_dot*l2*np.cos(teta_rad_inv1+teta_rad_inv2)-t2_dot**2*(l2*np.cos(teta_rad_inv1+teta_rad_inv2)))]])

    t1_ddot=((np.sin(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2))*(-r*np.cos(t)-(t1_dot**2*(-l1*np.sin(teta_rad_inv1)-l2*np.sin(teta_rad_inv1+teta_rad_inv2))-2*t1_dot*t2_dot*l2*np.sin(teta_rad_inv1+teta_rad_inv2)-t2_dot**2*(l2*np.sin(teta_rad_inv1+teta_rad_inv2)))))+(((-np.cos(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2)))*(-r*np.sin(t)-(t1_dot**2*(l1*np.cos(teta_rad_inv1)-l2*np.cos(teta_rad_inv1+teta_rad_inv2))-2*t1_dot*t2_dot*l2*np.cos(teta_rad_inv1+teta_rad_inv2)-t2_dot**2*(l2*np.cos(teta_rad_inv1+teta_rad_inv2)))))
    t2_ddot=((-(l1*np.sin(teta_rad_inv1)+l2*np.sin(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2))*(-r*np.cos(t)-(t1_dot**2*(-l1*np.sin(teta_rad_inv1)-l2*np.sin(teta_rad_inv1+teta_rad_inv2))-2*t1_dot*t2_dot*l2*np.sin(teta_rad_inv1+teta_rad_inv2)-t2_dot**2*(l2*np.sin(teta_rad_inv1+teta_rad_inv2)))))+(((l1*np.cos(teta_rad_inv1)+l2*np.cos(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2))*(-r*np.sin(t)-(t1_dot**2*(l1*np.cos(teta_rad_inv1)-l2*np.cos(teta_rad_inv1+teta_rad_inv2))-2*t1_dot*t2_dot*l2*np.cos(teta_rad_inv1+teta_rad_inv2)-t2_dot**2*(l2*np.cos(teta_rad_inv1+teta_rad_inv2)))))

    '''State variables'''#Variables de estado
    x = np.zeros((n, 4))

    '''Control vector'''#Señales de control
    u = np.zeros(( 2,n))


    '''Initial conditions'''#Condiciones iniciales
    x[0, 0] =np.pi/2# Initial pendulum position 1 (rad)
    x[0, 1] =0# Initial pendulum position 2 (rad)
    x[0, 2]=0 # Initial pendulum velocity (rad/s)
    x[0, 3]=0 # Initial pendulum velocity (rad/s)

    th1_ddot=np.zeros(n)

    ie_th1 = 0
    ie_th2 = 0

    '''State equation'''#Ecuacion de estado
    xdot = [0, 0, 0, 0]

    '''Dynamic simulation'''
    for i in range(n - 1):
        '''Current states'''
        th1 = x[i, 0]
        th2 = x[i, 1]
        th1_dot=x[i,2]
        th2_dot=x[i,3]
    
        '''Controller'''
        M=np.array([[(m1*lc1**2)+I1+I2+m2*((l1**2)+(lc2**2)+(2*l1*lc2*np.cos(th2))),(m2*lc2**2)+I2+m2*l1*lc2*np.cos(th2)],[(m2*lc2**2)+I2+m2*l1*lc2*np.cos(th2), (m2*lc2**2)+I2]])
        #Fuerzas centrípeta y de Coriolis
        C=np.array([[-2*m2*l1*lc2*th2_dot*np.sin(th2) +b1 ,-m2*l1*lc2*np.sin(th2)*th2_dot],[m2*l1*lc2*th1_dot*np.sin(th2) , b2]])
        #Aporte gravitacional
        gra=np.array([[m1*lc1*gravi*np.sin(th1)+m2*gravi*(l1*np.sin(th1)+lc2*np.sin(th1+th2))],[m2*lc2*gravi*np.sin(th1+th2)]])
    
        e_th1 = teta_rad_inv1[i] - th1
        e_th1_dot =t1_dot[i]- th1_dot
    
        e_th2 = teta_rad_inv2[i]- th2
        e_th2_dot =t2_dot[i]- th2_dot

        Kp =h[0]#5#10 #3.60614907877409#5.7255997347206#10
        Kd =h[1]#10#0.973324679473922#5 #0.503359674635035#1.96901831751399#5
        #Ki = 
    
        Kp2 =h[2]#4.93017386912806#5#3.60614907877409#5.7255997347206#5
        Kd2 =h[3]#5#0.347734270091561#0.1#0.503359674635035#0.5554397672254#0.1
        #Ki2 = 0

        u[0,i] = Kp * e_th1 + Kd * e_th1_dot +M[0,0]*t1_ddot[i]+M[0,1]*t2_ddot[i]+C[0,0]*t1_dot[i]+C[0,1]*t2_dot[i]+gra[0,0]
        u[1,i] = Kp2 * e_th2 + Kd2 * e_th2_dot +M[1,0]*t1_ddot[i]+M[1,1]*t2_ddot[i]+C[1,0]*t1_dot[i]+C[1,1]*t2_dot[i]+gra[1,0]
    
        '''Propiedades del modelo dinámico'''
        #Efecto inercial
   
        v=np.array([[th1_dot],[th2_dot]])
        C2=np.dot(C,v)
        ua=np.array([[u[0,i]],[u[1,i]]])
        aux1=ua-C2-gra
        Minv=linalg.inv(M)
        aux2=np.dot(Minv,aux1)
        xdot[0] = th1_dot
        xdot[1]= th2_dot
        xdot[2]=aux2[0,:]
        th1_ddot[i]=xdot[3]
        xdot[3]=aux2[1,:]
        '''Integrate dynamics'''
        x[i + 1, 0] = x[i, 0] + xdot[0] * dt
        x[i + 1, 1] = x[i, 1] + xdot[1] * dt
        x[i + 1, 2] = x[i, 2] + xdot[2] * dt
        x[i + 1, 3] = x[i, 3] + xdot[3] * dt
    
        ie_th1 = ie_th1 + e_th1 * dt
        ie_th2 = ie_th2 + e_th2 * dt
        ise=0
        iadu=0
        ise_next=0
        iadu_next=0
    
        ise=ise_next+(e_th1**2)*dt+(e_th2**2)*dt
        iadu=iadu_next+ (abs(u[0,i]-u[0,i-1]))*dt+(abs(u[1,i]-u[1,i-1]))*dt
        g=0
        if(ise>=10):
            ie=10
            g+=1
        else:
                ie=ise
                g+=0
        if(iadu>=1):
            ia=1
            g+=1
        else:
            ia=iadu
            g+=0
        if(g==2):
            print(g)
   
        ise_next=ie
        iadu_next=ia
    print(ise_next)
    print(iadu_next)
    
    return np.array([ise_next, iadu_next]),g

p=np.array([9.91102555, 4,5 ,0.1])
double_pendulum(p)

