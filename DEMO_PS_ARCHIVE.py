# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:16:28 2022

@author: marco
"""

import random
import numpy as np 
import math
import matplotlib.pyplot as plt
from drawnow import *

#---------------------Parametros DE-----------------------------#
limit=[(0,10),(0,10),(0,10)]       # Limites inferior y superior
poblacion = 200                    # Tamaño de la población, mayor >= 4
f_mut = 0.5                        # Factor de mutacion [0,2]
recombination = 0.7                # Tasa de  recombinacion [0,1]
generaciones =10                 # Número de generaciones
D = 3                              # Dimensionalidad O número de variables de diseño 
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
#Funcón de límite del actuador

def limcontro(u):
    if(u>2.94):
        ur=2.94
    elif(u>=-2.94 and u<=2.94):
        ur=u
    else:
        ur=-2.94
    
    return ur
#-----------------------------------------------------------------------------

pardps=np.array([0.5,1,0.05,0.006])
#----------Problema de optimización---------
def pendulum_s(r):
    '''Time Parameters'''
    dt = 0.005  # Tiempo de muestreo (5ms)
    ti = 0.0  # Tiempo inicial de la simulación (0s)
    tf = 10.0  # Tiempo inicial de la simulación (10s)
    n = int((tf - ti) / dt) + 1  # Número de muestras
    t = np.linspace(ti, tf, n)  # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)
    
    '''Dynamics Parameters'''
    m = 0.5  # Masa del pendulo (kg)
    l = 1.0  # Longitud de la barra del péndulo (m)
    lc = 0.3  # Longitud al centro de masa del péndulo (m)
    b = 0.05  # Coeficiente de fricción viscosa pendulo
    g = 9.81  # Aceleración de la gravedad en la Tierra
    I = 0.006  # Tensor de inercia del péndulo

    '''State variables'''
    x = np.zeros((n, 2))

    '''Control vector'''
    u = np.zeros((n, 1))
    
    
    ise=0
    ise_next=0
    iadu=0
    iadu_next=0
    
    '''Initial conditions'''
    x[0, 0] = 0  # Initial pendulum position (rad)
    x[0, 1] = 0  # Initial pendulum velocity (rad/s)
    ie_th = 0

    '''State equation'''
    xdot = [0, 0]

    '''Dynamic simulation'''
    for o in range(n - 1):
        '''Current states'''
        th = x[o, 0]
        th_dot = x[o, 1]
        e_th =np.pi-th
        e_th_dot = 0 - th_dot
        
        '''Controller'''
        Kp =r[0]
        Kd =r[1]
        Ki =r[2]
        
        u[o,0]= limcontro(Kp * e_th + Kd * e_th_dot + Ki * ie_th)
 
        
        
        '''System dynamics'''
        
        xdot[0] = th_dot
        xdot[1] = (u[o] - m * g * lc * np.sin(th) - b * th_dot) / (m * lc ** 2 + I)
        
        '''Integrate dynamics'''
        x[o + 1, 0] = x[o, 0] + xdot[0] * dt
        x[o + 1, 1] = x[o, 1] + xdot[1] * dt
        ie_th = ie_th + e_th * dt
        
        ise=ise_next+(e_th**2)*dt
        iadu=iadu_next+ (abs(u[o]-u[o-1]))*dt
        g=0
        if(ise>=2):
            ie=2
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
        #print(u[o,0])
       
    
    return np.array([ise_next, iadu_next]),g
#----------------------------------------------------------------------------------------------------


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
        
           
            
            v_mutante = x_1 + f_mut * (x_2 - x_3)
            print(v_mutante)
            break
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
    print(a)
    print(f_a)
    #-------Guardar en archivo excel-----------------------------------------
  
    filename="afa.csv" 
    myFile=open(filename,'w') 
    myFile.write("kp,kd,ki,f1, f2 \n") 
    for l in range(len(f_a)): 
        myFile.write(str(a[l, 0])+","+str(a[l, 1])+","+str(a[l, 2])+","+str(f_a[l, 0])+","+str(f_a[l, 1])+"\n") 
    myFile.close()
    #------------Gráfica del Frente de Pareto-----------------------
    plt.figure(1)
    plt.title('Aproximacion al frente de Pareto')
    plt.scatter(f_a[:, 0], f_a[:, 1])
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()
    
    return f_a

#llamado de la función main de DE
var=main(pendulum_s, limit, poblacion, f_mut, recombination, generaciones)