# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 23:15:18 2022

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


#---------------------Parametros DE-----------------------------#
limit=[[0,8],[0,5],[0,5],[0,5]]       # Limites inferior y superior
poblacion = 200                    # Tamaño de la población, mayor >= 4
f_mut = 0.5                        # Factor de mutacion [0,2]
recombination = 0.7                # Tasa de  recombinacion [0,1]
generaciones =  4               # Número de generaciones
D = 4                             # Dimensionalidad O número de variables de diseño 
M = 2                              # Numero de objetivos
AMAX = 30                          # Numero maximo de soluciones en el archivo
#----------------------------------------------------------------

#---------------Función de dominancia------#
def dominates(_a, _b):
    for _j in range(M):               #Recorre el vector J de funciones objetivo
        if _b[_j] < _a[_j]:    
            return False              #Regresa False si a domina b, en este caso seleccionamos b
    return True                       #Regresa Trux si b domina a, en este caso seleccionamos a
#----------------------------------------------------------------------------------------------------
def double_pendulum(h):
    #print(h)
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

    teta_rad_inv1= np.arctan2(Xp,-Yp)-np.arctan2(l2*np.sin(teta_rad_inv2),(l1+l2*np.cos(teta_rad_inv2)))

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

    t1_dot=(((np.sin(teta_rad_inv1+teta_rad_inv2))/(l1*np.sin(teta_rad_inv2)))*dx)-((np.cos(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2))*dy)
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

    ise=0
    iadu=0
    ise_next=0
    iadu_next=0

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
        #th1_ddot[i]=xdot[3]
        xdot[3]=aux2[1,:]
        '''Integrate dynamics'''
        x[i + 1, 0] = x[i, 0] + xdot[0] * dt
        x[i + 1, 1] = x[i, 1] + xdot[1] * dt
        x[i + 1, 2] = x[i, 2] + xdot[2] * dt
        x[i + 1, 3] = x[i, 3] + xdot[3] * dt
    
        # ie_th1 = ie_th1 + e_th1 * dt
        # ie_th2 = ie_th2 + e_th2 * dt
    
        ise=ise_next+(e_th1**2)*dt+(e_th2**2)*dt
        iadu=iadu_next+ (abs(u[0,i]-u[0,i-1]))*dt+(abs(u[1,i]-u[1,i-1]))*dt
        g=0
        if(ise>=20):
            ie=20
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
        # if(g==2):
        #     print(g)
   
        ise_next=ie
        iadu_next=ia
    # print(ise_next)
    # print(iadu_next)
    
    return np.array([ise_next, iadu_next]),g

#---------------Asegurar limites de caja-------------------------------------------------------------
def asegurar_limites(vec, limit):

    vec_new = []
    # ciclo que recorren todos los individuos
    for i in range(len(vec)):

        # Si el individuo sobrepasa el limite mínimo
        if vec[i] < limit[i][0]:
            vec_new.append(limit[i][0])

        # Si el individuo sobrepasa el limite máximom
        if vec[i] > limit[i][1]:
            vec_new.append(limit[i][1])

        # Si el individuo está dentro de los límites 
        if limit[i][0] <= vec[i] <= limit[i][1]:
            vec_new.append(vec[i])
        
    return vec_new
#---------------------------------------------------------------------------------------------------

def main(function, limites, poblacion, f_mut, recombination, generaciones):
    
    #-----Poblacion------------------------------------------------------------#
    population =  np.zeros((generaciones,poblacion, D)) #poblacion actual
    population_next= np.zeros((generaciones,poblacion, D)) #poblacion siguiente 
    #---------------------------------------------------------------------------

    #------------------F(x)---------------------------------------------------#
    f_x = np.zeros((generaciones,poblacion, M))  # Valor de funcion objetivo de poblacion actual
    f_x_next = np.zeros((generaciones,poblacion, M))  # Valor de funcion objetivo de poblacion siguiente
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    g_x = np.zeros((generaciones,poblacion))  # Valor de violacion de restricciones de poblacion actual
    g_x_next = np.zeros((generaciones,poblacion))  # Valor de violacion de restricciones de poblacion siguiente
    a = np.empty((0, D))  # Archivo
    f_a = np.empty((0, M))  # Valor de funcion objetivo para cada elemento del archivo
    g_a = np.empty(0)  # Valor de funcion objetivo para cada elemento del archivo
    #---------------------------------------------------------------------------
    
    # #--------------------Inicialización de la población-------------------------
    # for i in range(0,poblacion): # cambiar tam_poblacion
    #     indv = []
    #     for j in range(len(limites)):
    #         #print(len(limites)
    #         population[0][i][j]=random.uniform(limites[j][0],limites[j][1])
    #         population_next[0][i][j]=random.uniform(limites[j][0],limites[j][1])
    # #-------------------------------------------------------------------------------
    li=np.array(limites)
    population[0]=li[:,0] + np.random.rand(poblacion, D) * (li[:,1] - li[:,0])  # Inicializa poblacion
    population_next[0]=li[:,0] + np.random.rand(poblacion, D) * (li[:,1] - li[:,0])

    #-------------Evaluación población 0------------------------------------------------------------------
    for i, xi in enumerate(population[0,:]):  # Evalua objetivos
    
        f_x[0][i], g_x[0][i] = function(xi)
    #------------------------------------------------------------------------------------------------------

    #---------------------Ciclo evolutivo------------------------------------------------------------------
    for i in range(0,generaciones-1):
        print ('Generación:',i) 
        for j in range(0, poblacion):
            
            #Mutacion 
            # Seleccionamos 4 posiciones de vector aleatorios, range = [0, poblacion)
            # candidatos = range(0,poblacion)
            # print()
            # random_index = random.sample(candidatos, 4)
            
            r1 = j
            r2 = j
            r3 = j

            while r1 == j:
                r1 = random.randint(0, poblacion-1)

            while r2 == r1 or r2 == i:
                r2 = random.randint(0, poblacion-1)

            while r3 == r2 or r3 == r1 or r3 == i:
                r3 = random.randint(0, poblacion-1)
        
        
            x_1 = population[i][r1]
            x_2 = population[i][r2]
            x_3 = population[i][r3]
            x_t = population[i][j]
        
            # # Restamos x3 de x2, y creamos un nuevo vector (x_diff)
            # x_diff =[x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

            # # Multiplicamos x_diff por el factor de mutacion(F) y sumamos x_1
            # v_mutante =   [x_1_i + f_mut * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            
            v_mutante = x_1 + f_mut * (x_2 - x_3)
            v_mutante = asegurar_limites(v_mutante, limites)
            #print(v_mutante)
            #Vector hijo
            v_hijo =  np.copy(population[i][j])
            jrand = random.randint(0, D)
            
            for k in range(len(x_t)):
                
                crossover = random.uniform(0, 1)
                if crossover <= recombination or  k == jrand:
                    v_hijo[k]=v_mutante[k]
                else:
                    v_hijo[k]=x_t[k]
                    
            # Evalua descendiente
            f_ui, g_ui = function(v_hijo)

            #-------------------------Caso 1-----------------------------------------
            flag_ui=True
            if g_ui == 0 and g_x[i][j] == 0:
                # Selecciona el individuo que pasa a la siguiente generacion
                if dominates(f_ui, f_x[i][j]):
                    flag_ui=True
                elif dominates(f_x[i][j], f_ui):
                    flag_ui=False
                else:
                    if random.uniform(0, 1) < 0.5:
                        flag_ui=True
                    else:
                        flag_ui=False
            elif g_ui > g_x[i][j]:
                flag_ui=False
            elif g_ui < g_x[i][j]:
                flag_ui=True
            else:
                if random.uniform(0, 1) < 0.5:
                    flag_ui=True
                else:
                    flag_ui=False
            if flag_ui:
                f_x_next[i][j] = np.copy(f_ui)
                population_next[i][j] = np.copy(v_hijo)
                g_x_next[i][j]= np.copy(g_ui)
            else:
                f_x_next[i][j] = np.copy(f_x[i][j])
                population_next[i][j] = np.copy(population[i][j])
                g_x_next[i][j] = np.copy(g_x[i][j])
                

        # Una vez que termina la generacion actualizo x y f_x
        f_x[i+1] = np.copy(f_x_next[i])
        population[i+1] = np.copy(population_next[i])
        g_x[i+1] = np.copy(g_x_next[i])
        
    #-------------------------Archivo--------------------------------------------------------------------
        # Actualiza archivo (unicamente con soluciones factibles)
        for r, g_x_i in enumerate(g_x[i+1,:]):
            if g_x_i == 0:
                f_a = np.append(f_a, [f_x[i+1][r]], axis=0)
                a = np.append(a, [population[i+1][r]], axis=0)
        
        f_a_fil = np.empty((0, M))  # Conjunto no dominado
        a_fil = np.empty((0, D))  # Conjunto no dominado

        for i1, f_a_1 in enumerate(f_a):
            sol_nd = True
            for i2, f_a_2 in enumerate(f_a):
                if i1 != i2:
                    if dominates(f_a_2, f_a_1):
                        sol_nd = False
                        break
            if sol_nd:
                # f_x_fil.append(f_x_1)
                f_a_fil = np.append(f_a_fil, [f_a_1], axis=0)
                a_fil = np.append(a_fil, [a[i1]], axis=0)

        a = a_fil
        f_a = f_a_fil
        #print(a)
        #print(f_a)

        if len(a) > AMAX:
            # Ordenamiento del archivo con respecto a f1
            sorted_index = f_a[:, 0].argsort()
            f_a = f_a[sorted_index]
            a = a[sorted_index]

            # Calculo de distancias (crowding = apiñonamiento)
            distances = np.zeros(len(a))
            distances[0] = np.inf
            distances[-1] = np.inf

            for i in range(1, len(a) - 1):
                distances[i] += np.abs(f_a[i - 1, 0] - f_a[i + 1, 0]) + np.abs(
                f_a[i - 1, 1] - f_a[i + 1, 1])  # Crowding distance

            # Ordenamiento del archivo con respecto a las distancias
            sorted_index = distances.argsort()
            f_a = f_a[sorted_index]
            a = a[sorted_index]

            # Poda o depuracion del archivo (remueve las soluciones sobrantes del archivo)
            while len(a) != AMAX:
                a = np.delete(a, 0, 0)
                f_a = np.delete(f_a, 0, 0)
    
    #print(len(f_a))
    # #-------Guardar en archivo excel-----------------------------------------
  
    # filename="pdfa.csv" 
    # myFile=open(filename,'w') 
    # myFile.write("kp,kd,kp1,kd1,f1, f2 \n") 
    # for l in range(len(f_a)): 
    #     myFile.write(str(a[l, 0])+","+str(a[l, 1])+","+str(a[l, 2])+","+str(a[l, 3])+","+str(f_a[l, 0])+","+str(f_a[l, 1])+"\n") 
    # myFile.close()
    #------------Gráfica del Frente de Pareto-----------------------
    # plt.figure(1)
    # plt.title('Aproximacion al frente de Pareto')
    # plt.scatter(f_a[:, 0], f_a[:, 1])
    # plt.xlabel('f1')
    # plt.ylabel('f2')
    # plt.show()
    
    return f_a,a

#llamado de la función main de DE


Hvdemopd=np.zeros(30)

for k in range(30):
    print(k)
    var=main(double_pendulum, limit, poblacion, f_mut, recombination, generaciones)
    t=var[0]
    x=t[:,0]
    y=t[:,1]
   
    x = np.sort(x)
    y = np.sort(y)[::-1]
    x_max = 20
    y_max = 1
    yd=0
    area2=0
    for i in range(len(x)):
        if i == 0:  # primer elemento
            yd=0
            
            area2=0
            y_d=y_max-y[i]
            x_d2=x_max-x[i]
            area2=x_d2*y_d
        elif (0<i<len(x)-1):
            y_d=y[i-1]-y[i]
            x_d2=x_max-x[i]
            area2=area2+(y_d*x_d2)
        elif i == len(x)-1:  # ultimo elemento
            y_d=y[i-1]-y[i]
            x_d3=x_max-x[i-1]
            area2=area2+(y_d*x_d3)
            print('Hipervolumen:')
            print( area2)
        Hvdemopd[k]=area2
    
filename="Hvoldemopd.csv" 
myFile=open(filename,'w') 
myFile.write("Hv \n") 
for l in range(len(Hvdemopd)): 
    myFile.write(str(Hvdemopd[l])+"\n")  
myFile.close()