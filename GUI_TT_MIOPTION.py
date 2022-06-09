    # -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:17:44 2022

@author: marco
"""

import PySimpleGUI as sg
import math 
import numpy as np 
import random
import math
import matplotlib.pyplot as plt
from drawnow import *
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from numpy.linalg import inv
from matplotlib.widgets  import RectangleSelector
import os
from scipy import linalg
from scipy.integrate import odeint 
import time
import pylab as py
from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib import pyplot as plt
import mplcursors


#---------------------Parametros DEPS-----------------------------#
limit=[(0,10),(0,10),(0,10)]       # Limites inferior y superior
#poblacion = 200                    # Tamaño de la población, mayor >= 4
f_mut = 0.5                        # Factor de mutacion [0,2]
recombination = 0.7                # Tasa de  recombinacion [0,1]
#generaciones =10                   # Número de generaciones
D = 3                              # Dimensionalidad O número de variables de diseño 
M = 2                              # Numero de objetivos

#AMAX = 30                          # Numero maximo de soluciones en el archivo
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
    if(u > 2.94):
        ur = 2.94
    elif(u >= -2.94 and u <= 2.94):
        ur = u
    else:
        ur = -2.94
    
    return ur
#-----------------------------------------------------------------------------


#----------Problema de optimización---------
def pendulum_s(r,dyna):
    '''Time Parameters'''
    dt = 0.005  # Tiempo de muestreo (5ms)
    ti = 0.0  # Tiempo inicial de la simulación (0s)
    tf = 10.0  # Tiempo inicial de la simulación (10s)
    n = int((tf - ti) / dt) + 1  # Número de muestras
    t = np.linspace(ti, tf, n)  # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)
    kt=0.041042931
    R=4.19
    
    '''Dynamics Parameters'''
    m = dyna[0]  # Masa del pendulo (kg)
    l = dyna[1]  # Longitud de la barra del péndulo (m)
    lc = dyna[2]  # Longitud al centro de masa del péndulo (m)
    b = dyna[3]  # Coeficiente de fricción viscosa pendulo
    g = 9.81  # Aceleración de la gravedad en la Tierra
    I = dyna[4]  # Tensor de inercia del péndulo

    '''State variables'''
    x = np.zeros((n, 2))

    '''Control vector'''
    u = np.zeros((n, 1))
    vol=np.zeros(n)

    
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
        e_th =dyna[5]-th
        e_th_dot = 0 - th_dot
        
        '''Controller'''
        Kp =r[0]
        Kd =r[1]
        Ki =r[2]
        
        vol[o] =(limcontro(Kp * e_th + Kd * e_th_dot + Ki * ie_th)/(14*kt))+(th_dot*kt*14/R)
        #print(vol[o])
        u[o,0]=limcontro(((vol[o]/kt)-th_dot)*(kt**2/R)*131)
 
        
        
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
        if(ise>=10):
            ie=10
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
       
    
    return np.array([ise_next, iadu_next]),g,x,u,t
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

def main(function, limites, poblacion, f_mut, recombination, generaciones,pardyna,D,M,AMAX):
    D=D
    M=M
    AMAX=AMAX
    
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
    
   
    li=np.array(limites)
    population[0]=li[:,0] + np.random.rand(poblacion, D) * (li[:,1] - li[:,0])  # Inicializa poblacion
    population_next[0]=li[:,0] + np.random.rand(poblacion, D) * (li[:,1] - li[:,0])

    #-------------Evaluación población 0------------------------------------------------------------------
    for i, xi in enumerate(population[0,:]):  # Evalua objetivos
        solu=function(xi,pardyna)
        f_x[0][i], g_x[0][i] =solu[0],solu[1] #function(xi,pardyna)
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
            
            solu1=function(v_hijo,pardyna)
            f_ui, g_ui = solu1[0],solu1[1]
            
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

    
    return f_a,a
#-------Algoritmo génetico y operadores------

#---------------------------------Selección elitista------------------------------------------
def selecga(f, g, po, D, M):
    pop_x_r = np.empty((0, D))
    f_x_r = np.empty((0, M))
    g_x_r = np.empty(0)

    for r, f_x_i in enumerate(f):
        sol_nd = True
        g_x_i=g[r]
        
        for i2, f_a_2 in enumerate(f):
            if r != i2 and g_x_i==0:
                if dominates(f_a_2, f_x_i):
                    sol_nd = False
                    break
        if sol_nd:
            f_x_r = np.append(f_x_r, [f[r]], axis=0)
            pop_x_r = np.append(pop_x_r, [po[r]], axis=0)
            g_x_r = np.append(g_x_r, [g[r]], axis=0)

    return f_x_r, pop_x_r, g_x_r
#-------------------------SBX------------------------------------------
def crossov(p1,p2,eta,llo,lup):
    esp=1e-14
    
    for i, (x1, x2) in enumerate(zip(p1, p2)):
        rand = random.random()
        if rand <= 0.5:
            if(abs(p1[i]-p2[i])>esp):
                if(p1[i]<p2[i]):
                    y1=p1[i]
                    y2=p2[i]
                else:
                    y1=p2[i]
                    y2=p1[i]
                lb=llo[i]
                up=lup[i]
                ran = random.random()
                beta =1.0+ ((2. *(y1-lb))/(y2-y1))
                alpha = 2.0 - ((beta)**(-(eta + 1.0)))
                if (ran <= (1.0 / alpha)):
                    betaq = (ran * alpha)**((1.0 / (eta + 1.0)))
                else:
                    betaq = (1.0 / (2.0 - ran * alpha))**(1.0 / (eta + 1.0))
                c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                beta = 1.0 + (2.0 * (up - y2) / (y2 - y1))
                alpha = 2.0 - (beta)**(-(eta + 1.0))
                
                if (ran <= (1.0 / alpha)):
                    betaq = ((ran * alpha))**((1.0 / (eta + 1.0)))
                else:
                    betaq = (1.0 / (2.0 - ran * alpha))**( 1.0 / (eta + 1.0))
                c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))
                if (c1 > up):
                    c1 = up
                if (c1 < lb):
                    c1 = lb
                if (c2 > up):
                    c2 = up
                if (c2 < lb):
                    c2 = lb
                
                if (random.random() <= 0.5):
                   p1[i]=c2
                   p2[i]=c1
                else:
                    p1[i]=c1
                    p2[i]=c2
            else:
                p1[i]=p1[i]
                p2[i]=p2[i]
        else:
            p1[i]=p1[i]
            p2[i]=p2[i]
            
    return p1, p2
#-----------------Mutación Polinomial----------------------------------------------------
def mutPolynomial(individual, eta,lb,up,D):
    size = len(individual)
    pm=1/D

    for i in range(size):
        for k in range(D):
            if random.random() <= pm:
                x = individual[i][k]
                yl = lb[i][k]
                yu=up[i][k]
                if yl == yu:
                    x = yl
                else:
                    delta1 = (x - yl) / (yu - yl)
                    delta2 = (yu - x) / (yu - yl)
                    ra = random.random()
                    mut_pow = 1.0 / (eta + 1.)
                    if ra <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * ra + (1.0 - 2.0 * ra) * ((xy)**( eta + 1.0))
                        delta_q = (val)**( mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - ra) + 2.0 * (ra - 0.5) * ((xy)**(eta + 1.0))
                   
                        delta_q = 1.0 - (val)**(mut_pow)
                    x = x + delta_q *(yu - yl)
                    individual[i][k] = x
    return np.array(individual)

#--------------- Algoritmo génetico
def moga( limites, pop,eta, gen,D,M,AMAX,function,pardyna):
    D=D
    M=M
    AMAX=AMAX
    print(D)
    #-----Poblacion------------------------------------------------------------#
    population =  np.zeros((gen,pop, D)) #poblacion actual
    population_next= np.zeros((gen,pop, D)) #poblacion siguiente 
    #---------------------------------------------------------------------------
    #------------------F(x)---------------------------------------------------#
    f_x = np.zeros((gen,pop, M))  # Valor de funcion objetivo de poblacion actual
    f_x_next = np.zeros((gen,pop, M))  # Valor de funcion objetivo de poblacion siguiente
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    g_x = np.zeros((gen,pop))  # Valor de violacion de restricciones de poblacion actual
    g_x_next = np.zeros((gen,pop))  # Valor de violacion de restricciones de poblacion siguiente
    a = np.empty((0, D))  # Archivo
    f_a = np.empty((0, M))  # Valor de funcion objetivo para cada elemento del archivo
    g_a = np.empty(0)  # Valor de funcion objetivo para cada elemento del archivo
    #---------------------------------------------------------------------------
    
    li=np.array(limites)
  
    population[0]=li[:,0] + np.random.rand(pop, D) * (li[:,1] - li[:,0])  # Inicializa poblacion
    population_next[0]=li[:,0] + np.random.rand(pop, D) * (li[:,1] - li[:,0])
    
    #-------------Evaluación población 0----------------------------------------------------------------
    for i, xi in enumerate(population[0,:]):  # Evalua objetivos
        solu=function(xi,pardyna)
        f_x[0][i], g_x[0][i] =solu[0],solu[1] #function(xi,pardyna)
        #------------------------------------------------------------------------------------------------
    for i in range(0,gen-1):
        f_x_next[i][:]=f_x[i][:]
        population_next[i][:]=population[i][:]
        g_x_next[i][:]=g_x[i][:]
    
        #print ('Generación:',i) 
        selecc=selecga(f_x[i,:],g_x[i,:],population[i],D,M)
        f_x_s=selecc[0]
        popu_x_s=selecc[1]
        g_x_s=selecc[2]
    
        cross=[]
        if len(f_x_s) % 2 != 0:
            r1 = random.randint(0, len(popu_x_s)-1)
            p1=popu_x_s[r1,:]
        lb=np.zeros((len(f_x_s),D))
        up=np.ones((len(f_x_s),D))   
        for j in range(math.floor(len(popu_x_s)/2)):
        
            r1=j
            r2=j
            while r1 == j:
                r1 = random.randint(0, len(popu_x_s)-1)

            while r2 == r1 or r2 == j:
                r2 = random.randint(0, len(popu_x_s)-1)
            p1=popu_x_s[r1,:]
            p2=popu_x_s[r2,:]
        
          
            c=crossov(p1,p2,eta,lb[j],up[j])
            cross.append(c[0])
            cross.append(c[1])
        cro=np.array(cross)
        mut=mutPolynomial(cro,1,lb,up,D)
        f_x_off=np.zeros((len(mut),M))
        g_x_off=np.zeros(len(mut))
        
        for r in range(len(mut)):
            mut[r]=asegurar_limites(mut[r],limites)
            val=function(mut[r],pardyna)
            f_x_off[r]=val[0]
            g_x_off[r]=val[1]
        
            #-------------------------Caso 1-----------------------------------------
            flag_ui=True
            if g_x_off[r] == 0 and g_x[i][r] == 0:
                # Selecciona el individuo que pasa a la siguiente generacion
                if dominates(f_x_off[r], f_x[i][r]):
                    flag_ui=True
                elif dominates(f_x[i][r], f_x_off[r]):
                    flag_ui=False
                else:
                    if random.uniform(0, 1) < 0.5:
                        flag_ui=True
                    else:
                        flag_ui=False
            elif g_x_off[r] > g_x[i][r]:
                flag_ui=False
            elif g_x_off[r] < g_x[i][r]:
                flag_ui=True
            else:
                if random.uniform(0, 1) < 0.5:
                    flag_ui=True
                else:
                    flag_ui=False
            if flag_ui:
                f_x_next[i][r] = np.copy(f_x_off[r])
                population_next[i][r] = np.copy(mut[r])
                g_x_next[i][r]= np.copy(g_x_off[r])
            else:
                f_x_next[i][r] = np.copy(f_x[i][r])
                population_next[i][r] = np.copy(population[i][r])
                g_x_next[i][r] = np.copy(g_x[i][r])

        # Una vez que termina la generacion actualizo x y f_x
        f_x[i+1] = np.copy(f_x_next[i])
        population[i+1] = np.copy(population_next[i])
        g_x[i+1] = np.copy(g_x_next[i])
        
        #-------------------------Archivo--------------------------------------------------------------------
        # Actualiza archivo (unicamente con soluciones factibles)
        for k, g_x_i in enumerate(g_x[i+1,:]):
            if g_x_i == 0:
                f_a = np.append(f_a, [f_x[i+1][k]], axis=0)
                a = np.append(a, [population[i+1][k]], axis=0)
        
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
                f_a_fil = np.append(f_a_fil, [f_a_1], axis=0)
                a_fil = np.append(a_fil, [a[i1]], axis=0)
            
        a = a_fil
        f_a = f_a_fil

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
   
    return f_a,a

#-------------------------------------------

#------------------ PSO -----------------------------------------------------------------------------
def s_best(x, f, g, xbest, fxbest, gxbest, D, M, pop):
    x_b = np.zeros((pop, D))
    f_xb = np.zeros((pop, M))
    g_xb = np.zeros(pop)
    ui_is_better = True
    for u in range(pop):

        if g[u] == 0 and gxbest[u] == 0:  # Ambas soluciones son factibles
            if dominates(f[u], fxbest[u]):
                ui_is_better = True
            elif dominates(fxbest[u], f[u]):
                ui_is_better = False
            else:
                if random.uniform(0, 1) < 0.5:
                    ui_is_better = True
                else:
                    ui_is_better = False
        elif g[u] > gxbest[u]:  # Hijo viola mas restricciones que padre
            ui_is_better = False
        elif g[u] < gxbest[u]:  # Padre viola mas restricciones que hijo
            ui_is_better = True
        else:  # Ambos violan la misma cantidad de restricciones
            if random.uniform(0, 1) < 0.5:
                ui_is_better = True
            else:
                ui_is_better = False

        if ui_is_better:
            f_xb[u] = np.copy(f[u])
            g_xb[u] = np.copy(g[u])
            x_b[u] = np.copy(x[u])
        else:
            f_xb[u] = np.copy(fxbest[u])
            g_xb[u] = np.copy(gxbest[u])
            x_b[u] = np.copy(xbest[u])
    return x_b, f_xb, g_xb

# ---------------------------------------------------------------------------------------------------


def selecpso(f, g, po, D, M):
    pop_r = np.empty((0, D))
    f_x_r = np.empty((0, M))
    g_x_r = np.empty(0)

    best = 0
    for i in range(1, len(f)):
        if dominates(f[i], f[best]) and g[i] <= g[best]:
            best = i
    # print(best)
    pop_x_r = po[best]
    f_x_r = f[best]
    g_x_r = g[best]
    return f_x_r, pop_x_r, g_x_r

#-----------END FUNCIONES PSO----#


def MOPSO(function, limit, pop, Vmax, Vmin, c1, c2, gen, pardyna, D, M, AMAX):
    #-----Poblacion------------------------------------------------------------#
    population = np.zeros((gen, pop, D))  # poblacion actual
    population_next = np.zeros((gen, pop, D))  # poblacion siguiente
    x_best = np.zeros((gen, pop, D))  # Matriz de mejores posiciones locales
    x_best_swarp = np.zeros(D)  # Gbest
    # -----------------------------------------#

    #------------------F(x)---------------------------------------------------#
    # Valor de funcion objetivo de poblacion actual
    f_x = np.zeros((gen, pop, M))
    # Valor de funcion objetivo de poblacion siguiente
    f_x_next = np.zeros((gen, pop, M))
    f_x_best = np.zeros((gen, pop, M))  # Valor de función objetivo
    f_x_best_swarp = np.zeros(D)  # Gbest
    # -----------------------------------------#

    g_x = np.zeros((gen, pop))
    # Valor de violacion de restricciones de poblacion siguiente
    g_x_next = np.zeros((gen, pop))
    g_x_best = np.zeros((gen, pop))  # Valor de restricciones
    g_x_best_swarp = np.zeros(D)  # Mejor posición global
    a = np.empty((0, D))  # Archivo
    # Valor de funcion objetivo para cada elemento del archivo
    f_a = np.empty((0, M))
    # Valor de funcion objetivo para cada elemento del archivo
    g_a = np.empty(0)


# ---------------------------------------------------------------------------
    vel = np.zeros((pop, D))  # VECTOR DE VELOCIDADES
    li = np.array(limit)
    # Inicializa poblacion
    population[0] = li[:, 0] + np.random.rand(pop, D) * (li[:, 1] - li[:, 0])
    x_best[0] = population[0]


# -------------Evaluación población 0------------------------------------------------------------------
    for i, xi in enumerate(population[0, :]):  # Evalua objetivos
        solu = function(xi, pardyna)
        f_x[0][i], g_x[0][i] = solu[0], solu[1]  # function(xi,pardyna)
        # ------------------------------------------------------------------------------------------------------
    f_x_best[0] = f_x[0]
    g_x_best[0] = g_x[0]

    selecc = selecpso(f_x[0, :], g_x[0, :], population[0], D, M)
    f_x_best_swarp = selecc[0]
    x_best_swarp = selecc[1]
    g_x_best_swarp = selecc[2]

    for i in range(0, gen-1):

        print('Generación:', i)
        w = Vmax-(i / gen)*(Vmax-Vmin)
        r1 = random.random()
        r2 = random.random()
        for j in range(pop):
            vel[j] = w*vel[j] + r1*c1 * \
                (x_best[i][j]-population[i][j])+r2 * \
                c2*(x_best_swarp-population[i][j])
        population_next[i] = population[i]+vel
        for h in range(pop):
            population_next[i][h] = asegurar_limites(
                population_next[i][h], limit)
            sol = function(population_next[i][h], pardyna)
            f_x_next[i][h], g_x_next[i][h] = sol[0], sol[1]
        population[i+1] = population_next[i]
        f_x[i+1] = f_x_next[i]
        g_x[i+1] = g_x_next[i]
        sele = s_best(population[i+1], f_x[i+1], g_x[i+1],
                      x_best[i], f_x_best[i], g_x_best[i], D, M, pop)
        x_best[i+1] = sele[0]
        f_x_best[i+1] = sele[1]
        g_x_best[i+1] = sele[2]
        selecc = selecpso(f_x_best[i+1], g_x_best[i+1], x_best[i+1], D, M)
        f_x_best_swarp = selecc[0]
        x_best_swarp = selecc[1]
        g_x_best_swarp = selecc[2]

        '''ARCHIVO'''
        # Actualiza archivo (unicamente con soluciones factibles)
        for r, g_x_i in enumerate(g_x_best[i+1]):
            if g_x_i == 0:
                f_a = np.append(f_a, [f_x_best[i+1][r]], axis=0)
                a = np.append(a, [x_best[i+1][r]], axis=0)

        # Filtrado no dominado para el archivo
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
    return f_a, a

#----------------------------------------------------------------------------------------------------

#-----------------Péndulo invertido----------------
limitpi=[(0,10),(0,5),(0,5),(0,10),(0,5),(0,5)]       # Limites inferior y superior
#poblacionpi = 200                    # Tamaño de la población, mayor >= 4
f_mutpi = 0.5                        # Factor de mutacion [0,2]
recombinationpi = 0.7                # Tasa de  recombinacion [0,1]
#generacionespi =5                 # Número de generaciones
Dpi = 6                             # Dimensionalidad O número de variables de diseño 
Mpi = 2                              # Numero de objetivos
#AMAXpi = 30     

def inverted_pendulum(r,dimpi):
    
    '''Time parameters'''
    dt = 0.005  # Tiempo de muestreo (5ms)
    ti = 0.0  # Tiempo inicial de la simulación (0s)
    tf = 15.0  # Tiempo inicial de la simulación (10s)
    n = int((tf - ti) / dt) + 1  # Número de muestras
    t = np.linspace(ti, tf, n)  # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)
    
    '''Dynamic parameters'''
    m = dimpi[0]  # Masa del pendulo (kg)
    M = dimpi[1]  # Masa del carro (kg) 
    l = dimpi[2]  # Longitud de la barra del péndulo (m)
    lc = dimpi[3]  # Longitud al centro de masa del péndulo (m)
    b1 = dimpi[4]  # Coeficiente de fricción viscosa pendulo
    b2 = dimpi[5]  # Coeficiente de friccion del carro
    gra = 9.81  # Aceleración de la gravedad en la Tierra
    I = dimpi[6]  # Tensor de inercia del péndulo

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
    for c in range(n - 1):
        '''Current states'''
        x = z[c, 0]  # Posicion del carro
        th = z[c, 1]   # Posición del péndulo
        x_dot= z[c, 2] # Velocidad del carro
        th_dot = z[c, 3]  # Velocidad del péndulo

        '''Controller'''
        e_x = dimpi[8] - x #Error de posición de carro
        e_x_dot = 0 - x_dot #Error de velocidad de carro
        e_th = dimpi[7]-th #Error de posicón angular
        e_th_dot = 0 - th_dot #Error de velocidad angular 

        '''Ganancias del controlador del carro'''
        Kp = r[0]
        Kd = r[1]
        Ki = r[2]

        '''Ganancias del controlador del péndulo'''
        Kp1 =r[3]
        Kd1 =r[4]
        Ki1 =r[5]
        #print(r)

        u[0, c] = Kp * e_x + Kd * e_x_dot + Ki * ie_x #Señal de control del actuador del carro
        u[1, c] = Kp1 * e_th + Kd1 * e_th_dot + Ki1 * ie_th #Señal de control del actuador del péndulo
        
        # print(u[0,c])
        # print(u[1,c])

        MI = np.array([[M + m,-m * lc * np.sin(th)], [-m * lc * np.sin(th), I + m * lc ** 2]])  # Matriz de inercia
        MC = np.array([[b1, -m * lc * np.cos(th) * th_dot], [0, b2]])  # Matriz de Coriollis
        MG = np.array([[0], [m * gra * l * np.cos(th)]])  # Vector de gravedad

        array_dots = np.array([[x_dot], [th_dot]])#Vector de velocidades
        MC2 = np.dot(MC, array_dots)

        ua = np.array([[u[0, c]], [u[1, c]]])
        aux1 = ua - MC2 - MG
        Minv = inv(MI)
        aux2 = np.dot(Minv, aux1) #Varables de segundo grado /(doble derivada)

        '''System dynamics'''
        zdot[0] = x_dot #Velocidad del carro
        zdot[1] = th_dot #Velocidad del péndulo
        zdot[2] = aux2[0, :] #Aceleración del carro
        zdot[3] = aux2[1, :] #Aceleración del péndulo

        '''Integrate dynamics'''
        z[c + 1, 0] = z[c, 0] + zdot[0] * dt
        z[c + 1, 1] = z[c, 1] + zdot[1] * dt
        z[c + 1, 2] = z[c, 2] + zdot[2] * dt
        z[c + 1, 3] = z[c, 3] + zdot[3] * dt
        ie_th = ie_th + e_th * dt
        ie_x = ie_x+e_x*dt

        ise=ise_next+(e_th**2)*dt+(e_x**2)*dt
        iadu=iadu_next+ (abs(u[0,c]-u[0,c-1]))*dt+(abs(u[1,c]-u[1,c-1]))*dt
        g=0
        if(ise>=20):
            ie=20
            g+=1
        else:
            ie=ise
            g+=0
        if(iadu>=1.2):
            ia=1.2
            g+=1
        else:
            ia=iadu
            g+=0
   
        ise_next=ie
        iadu_next=ia
    u[:,n - 1] = u[:,n - 2] #Actualizar señal de control
    
    #print(z[:, 0])
    
    return np.array([ise_next, iadu_next]),g,z,u,t
 
#-----------------------------------------------------------------------------

#-----------Péndulo doble-----------------------------------------------------

#---------------------Parametros DE-----------------------------#
limitpd=[[0,8],[0,5],[0,5],[0,5]]       # Limites inferior y superior
#poblacionpd = 200                    # Tamaño de la población, mayor >= 4
f_mutpd = 0.5                        # Factor de mutacion [0,2]
recombinationpd = 0.7                # Tasa de  recombinacion [0,1]
#generacionespd =  4               # Número de generaciones
Dpd = 4                             # Dimensionalidad O número de variables de diseño 
Mpd = 2                              # Numero de objetivos
#AMAXpd = 30                          # Numero maximo de soluciones en el archivo
#----------------------------------------------------------------


def double_pendulum(h,dinde):
    #print(h)
    '''Time parameters''' #Parametros temporales
    dt = 0.005  # Tiempo de muestreo (5ms)
    ti = 0.0  # Tiempo inicial de la simulación (0s)
    tf =20  # Tiempo final de la simulación (12.25s)
    n = int((tf - ti) / dt) + 1  # Número de muestras
    t = np.linspace(ti, tf, n)  # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)
    
    '''Dynamic parameters''' #Parametros dinamicos
    m1 = dinde[0]  # Masa de la barra 1(kg)
    m2= dinde[1] #Masa de la barra 2 (kg)
    l1 = dinde[2] # Longitud de la barra 1 (m)
    lc1 =dinde[3]  # Longitud al centro de masa de la barra 2 (m)
    l2= dinde[4] #.0Longitud de la baraa 2 (m)
    lc2=dinde[5] #Longitud al centro de masa de la barra 2(m)
    b1 = dinde[6]  # Coeficiente de fricción viscosa de la barra 1
    b2= dinde[7] #Coeficiente de fricción viscosa de la barra 2
    gravi = 9.81  # Aceleración de la gravedad en la Tierra
    I1 = dinde[8]  # Tensor de inercia del péndulo 1
    I2= dinde[9]#Tensor de inercia del péndulo 2

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
    
    return np.array([ise_next, iadu_next]),g,x,u,t
#-----------------------------------------------------------------------------
#Function for drawing
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def pick_scatter_plot():
    # picking on a scatter plot (matplotlib.collections.RegularPolyCollection)

    #x, y, c, s = rand(4, 100)

    def onpick3(event):
        ind = event.ind
        print('onpick3 scatter:', ind, valu[ind, 3], valu[ind, 4])

    #fig, ax = plt.subplots()
    #ax.scatter(x, y, 100*s, c, picker=True)
    fig.canvas.mpl_connect('pick_event', onpick3)


def cursor1_annotations(sel):
    sel.annotation.set_text(
        'ISE: {:.4f} \nIADU: {:.4f}'.format(sel.target[0], sel.target[1]))

##-----DEFAULT SETTINGS----------------------------------##
bw: dict = {'size': (20, 20), 'font': ('Franklin Gothic Book', 60), 'button_color': ("blue", "#F8F8F8")}
bt: dict = {'size': (20, 1),'font': ('Franklin Gothic Book', 14,'bold italic'), }
bo: dict = {'size': (15, 2), 'font': ('Arial', 24), 'button_color': ("black", "#ECA527"), 'focus': True}

layouthome= [[sg.Text('CONTROL PID CON OPTIMIZACIÓN MULTIOBJETIVO',justification='center', 
             text_color='white', font=('Franklin Gothic Book', 28, 'bold'))],
             [sg.Text('Selecciona un péndulo:', justification='center',text_color='white', font=('Franklin Gothic Book', 14, 'bold'))],
             [sg.Button(image_filename='./img/PS.png' ,key='Simple',button_color=(sg.theme_background_color(), sg.theme_background_color())), sg.Button(image_filename='./img/PI.png', key='Invertido',button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/PD.png', key='Doble')],
             [sg.Text('Simple',size=(38,2),justification='center',font=('Franklin Gothic Book', 15, 'bold')), sg.Text('Invertido',size=(38,2), justification='center',font=('Franklin Gothic Book', 15, 'bold')),sg.Text('Doble',size=(38,2),justification='center',font=('Franklin Gothic Book', 15, 'bold'))],
             [sg.Button('Salir',button_color='red',size=(5,2),border_width=5,key='Exit')]]

layouts=[[sg.Text('Péndulo Simple:',text_color='white', font=('Franklin Gothic Book', 28, 'bold')) ],
            [sg.Text('Masa (m):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.Input('0.06555',key='masaps'),sg.Text('kg', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Longitud (l):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('0.443',key='lps'),sg.Text('m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Longitud al centro masa (lc):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('0.2215',key='lcps'),sg.Text('m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Fricción (b):', text_color='black', font=('Franklin Gothic Book', 12, 'bold '),size=(24,1)), sg.InputText('0.05',key='bps'),sg.Text('Ns/m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Momento de inercia (I):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('0.006',key='ips'),sg.Text('kgm^2', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Set point (rad):', text_color='black', font=('Franklin Gothic Book', 12, 'bold italic'),size=(24,1)), sg.InputText('3.1416',key='sps'),sg.Text('rad', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Seleccione el algoritmo metaheurístico:',text_color='white', font=('Franklin Gothic Book', 28, 'bold'))],
            [sg.Button('DE',button_color='blue',border_width=5,key='deps',**bt),sg.Button('GA',button_color='blue',border_width=5,key='geps',**bt),sg.Button('PSO',button_color='blue',border_width=5,key='psops',**bt)],
            [sg.Button(image_filename='./img/home.png', key='Homeps',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color()))]
            ]

layouti=[[sg.Text('Péndulo Invertido:',text_color='white', font=('Franklin Gothic Book', 28, 'bold')) ],
            [sg.Text('Masa del péndulo(m):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.Input('0.5',key='masapi'),sg.Text('kg', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Masa del carrito(M):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.Input('0.7',key='masaca'),sg.Text('kg', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Longitud (l):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('1.0',key='lpi'),sg.Text('m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Longitud al centro masa (lc):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('0.3',key='lcpi'),sg.Text('m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Fricción del péndulo (b1):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('0.05',key='bpi'),sg.Text('Ns/m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Fricción del carrito (b2):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('0.06',key='bca'),sg.Text('Ns/m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Momento de inercia (I):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('0.006',key='ipi'),sg.Text('kgm^2', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Set point del péndulo (rad):', text_color='black', font=('Franklin Gothic Book', 12, 'bold italic'),size=(24,1)), sg.InputText('1.57',key='spi'),sg.Text('rad', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Set point del carrito:', text_color='black', font=('Franklin Gothic Book', 12, 'bold italic'),size=(24,1)), sg.InputText('0',key='spc'),sg.Text('m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Seleccione el algoritmo metaheurístico:',text_color='white', font=('Franklin Gothic Book', 28, 'bold'))],
            [sg.Button('DE',button_color='blue',border_width=5,key='depi',**bt),sg.Button('GA',button_color='blue',border_width=5,key='gepipi',**bt),sg.Button('PSO',button_color='blue',border_width=5,key='psopi',**bt)],
            [sg.Button(image_filename='./img/home.png', key='Homepi',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color()))]
            ]

layoutd=[[sg.Text('Péndulo Doble:',text_color='white', font=('Franklin Gothic Book', 28, 'bold')) ],
            [sg.Text('Masa del brazo 1 (m1):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(28,1)), sg.Input('0.5',key='masapd1'),sg.Text('kg', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Masa del brazo 2 (m1):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(28,1)), sg.Input('0.5',key='masapd2'),sg.Text('kg', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Longitud del brazo 1 (l1):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(28,1)), sg.InputText('1',key='lpd1'),sg.Text('m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Longitud al centro masa 1 (lc1):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(28,1)), sg.InputText('0.5',key='lcpd1'),sg.Text('m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Longitud del brazo 2 (l2):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(28,1)), sg.InputText('1.0',key='lpd2'),sg.Text('m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Longitud al centro masa 2 (lc2):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(28,1)), sg.InputText('0.3',key='lcpd2'),sg.Text('m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Fricción del brazo 1 (b1):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(28,1)), sg.InputText('0.05',key='bpd1'),sg.Text('Ns/m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Fricción del brazo 2 (b2):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(28,1)), sg.InputText('0.02',key='bpd2'),sg.Text('Ns/m', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Momento de inercia  brazo 1 (I1):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(28,1)), sg.InputText('0.006',key='ipd1'),sg.Text('kg/m^2', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Momento de inercia  brazo 2 (I2):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(28,1)), sg.InputText('0.004',key='ipd2'),sg.Text('kgm^2', text_color='black', font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('Seleccione el algoritmo metaheurístico:',text_color='white', font=('Franklin Gothic Book', 28, 'bold'))],
            [sg.Button('DE',button_color='blue',border_width=5,key='depd',**bt),sg.Button('GA',button_color='blue',border_width=5,key='gepd',**bt),sg.Button('PSO',button_color='blue',border_width=5,key='psopd',**bt)],
            [sg.Button(image_filename='./img/home.png', key='Homepd',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color()))]
            ]

layoutde=[[sg.Text('Evolución diferencial',text_color='white', font=('Franklin Gothic Book', 28, 'bold'))],
          [sg.Text('Ingresa los siguientes  parámetros: ',text_color='white', font=('Franklin Gothic Book', 18, 'bold'))],
          [sg.Text('Tamaño de la población:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('200',key='popb')],
          [sg.Text('Número de generaciones:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('10',key='gen')],
          [sg.Text('Tamaño del archivo:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('30',key='Am')],
          [sg.Button(image_filename=r'./img/b1.png', key='rep',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homede',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Continuar',size=(10,2),border_width=5,key='con',button_color='blue')]
          ]
layoutga=[[sg.Text('Algoritmo génetico',text_color='white', font=('Franklin Gothic Book', 28, 'bold'))],
          [sg.Text('Ingresa los siguientes  parámetros: ',text_color='white', font=('Franklin Gothic Book', 18, 'bold'))],
          [sg.Text('Tamaño de la población:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('200',key='popga')],
          [sg.Text('Número de generaciones:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('10',key='genga')],
          [sg.Text('Tamaño del archivo:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('30',key='Amga')],
          [sg.Text('Eta :', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('1',key='eta')],
          [sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='repgaps',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homegaps',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Continuar',size=(10,2),border_width=5,key='congaps',button_color='blue')]],key='gepips'),
           sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='repgapi',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homegapi',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Continuar',size=(10,2),border_width=5,key='congapi',button_color='blue')]],key='gepi'),
           sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='repgapd',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homegapd',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Continuar',size=(10,2),border_width=5,key='congapd',button_color='blue')]],key='gepd')]
          ]

layoutpso=[[sg.Text('Algoritmo de optimización por enjambre de partículas (PSO)',text_color='white', font=('Franklin Gothic Book', 28, 'bold'))],
          [sg.Text('Ingresa los siguientes  parámetros: ',text_color='white', font=('Franklin Gothic Book', 18, 'bold'))],
          [sg.Text('Tamaño de la población:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('200',key='poppso')],
          [sg.Text('Número de generaciones:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('10',key='genpso')],
          [sg.Text('Tamaño del archivo:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('30',key='Ampso')],
          [sg.Text('Velocidad minima :', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('0',key='Vmin')],
          [sg.Text('Velocidad máxima :', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('0.1',key='Vmax')],
          [sg.Text('Alpha :', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('1',key='c1')],
          [sg.Text('Betha :', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('1',key='c2')],
          [sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='reppsops',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homepsops',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Continuar',size=(10,2),border_width=5,key='conpsops',button_color='blue')]],key='papsops'),
           sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='reppsopi',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homepsopi',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Continuar',size=(10,2),border_width=5,key='conpsopi',button_color='blue')]],key='papsopi'),
           sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='reppsopd',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homepsopd',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Continuar',size=(10,2),border_width=5,key='conpsopd',button_color='blue')]],key='papsopd')]
          ]



layoutdepi=[[sg.Text('Evolución diferencial',text_color='white', font=('Franklin Gothic Book', 28, 'bold'))],
          [sg.Text('Ingresa los siguientes  parámetros: ',text_color='white', font=('Franklin Gothic Book', 18, 'bold'))],
          [sg.Text('Tamaño de la población:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('200',key='popbpi')],
          [sg.Text('Número de generaciones:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('10',key='genpi')],
          [sg.Text('Tamaño del archivo:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('30',key='Ampi')],
          [sg.Button(image_filename=r'./img/b1.png', key='reppi',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homedepi',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Continuar',size=(10,2),border_width=5,key='conpi',button_color='blue')]
          ]

layoutdepd=[[sg.Text('Evolución diferencial',text_color='white', font=('Franklin Gothic Book', 28, 'bold'))],
          [sg.Text('Ingresa los siguientes  parámetros: ',text_color='white', font=('Franklin Gothic Book', 18, 'bold'))],
          [sg.Text('Tamaño de la población:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('200',key='popbpd')],
          [sg.Text('Número de generaciones:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('10',key='genpd')],
          [sg.Text('Tamaño del archivo:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(25,1)), sg.Input('30',key='Ampd')],
          [sg.Button(image_filename=r'./img/b1.png', key='reppd',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homedepd',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Continuar',size=(10,2),border_width=5,key='conpd',button_color='blue')]
          ]

tapsv=[['Espere','se está','ejecutando','código','......'],['Espera','estamos','ejecutando','código','......']]

layoutpfps=[[sg.Text('Seleccione un conjunto de ganancias del controlador  PID',text_color='white', font=('Franklin Gothic Book', 20, 'bold'))],
            [sg.Canvas(key='canw')],
            [sg.Table(values=tapsv ,headings=['Kp' , 'Kd' ,' Ki','ISE','IADU'],auto_size_columns=True,right_click_selects=True,enable_click_events=True, key='Tabl',vertical_scroll_only=False,num_rows=25 ), sg.Canvas(key='can')],
            [sg.Button('Simular',key='Simups'), sg.Button('Simular',key='Simupsga'),sg.Button('Simular',key='Simupspso')]]

tapiv=[['Espera','estamos','ejecutando','código','......','......','......','......'],['Espera','estamos','ejecutando','código','......','......','......','......']]
layoutpfpi=[[sg.Text('Seleccione un conjunto de ganancias del controlador  PID',text_color='white', font=('Franklin Gothic Book', 20, 'bold'))],
            [sg.Table(values=tapiv ,headings=['Kpcar' , 'Kdcar' ,'Kicar','Kppénd' , 'Kdpén' ,' Kipén','ISE','IADU'],auto_size_columns=True,right_click_selects=True,enable_click_events=True, key='Tablpi',vertical_scroll_only=False,num_rows=25 ), sg.Canvas(key='canpfpi')],
            [sg.Button('Simular',key='Simupi'), sg.Button('Simular',key='Simupiga'),sg.Button('Simular',key='Simupipso')]]
tapdv=[['Espera','estamos','ejecutando','código','......','......'],['Espera','estamos','ejecutando','código','......','......']]

layoutpfpd=[[sg.Text('Seleccione un conjunto de ganancias del controlador  PID',text_color='white', font=('Franklin Gothic Book', 20, 'bold'))],
            [sg.Table(values=tapdv ,headings=['Kp1' , 'Kd1' ,'Kp2' , 'Kd2' ,'ISE','IADU'],auto_size_columns=True,right_click_selects=True,enable_click_events=True, key='Tablpd',vertical_scroll_only=False,num_rows=25 ), sg.Canvas(key='canpfpd')],
            [sg.Button('Simular',key='Simupd'),sg.Button('Simular',key='Simupdga'),sg.Button('Simular',key='Simupdpso')]]


layoutsimpan=[[sg.Canvas(key='canani')]]
layoutsimpgra=[[sg.Canvas(key='cangraps')]]

layouttap=[[sg.TabGroup([[sg.Tab('Animación',layoutsimpan),sg.Tab('Gráficas', layoutsimpgra)]],tab_location='centertop',border_width=5)],
           [sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='Returnps',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homesimups',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Salir',button_color='red',size=(3,2),border_width=5,key='Exit0')]],key='paps'),
           sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='Returnpsga',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homesimupsga',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Salir',button_color='red',size=(3,2),border_width=5,key='Exit10')]],key='gapaps'),
           sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='Returnpspso',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homesimupspso',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Salir',button_color='red',size=(3,2),border_width=5,key='Exit20')]],key='psopaps')
           ]]

layoutinvpan=[[sg.Canvas(key='cananipi')]]
layoutinvpgra=[[sg.Canvas(key='cangrapi')]]

layouttappi=[[sg.TabGroup([[sg.Tab('Animación',layoutinvpan),sg.Tab('Gráficas', layoutinvpgra)]],tab_location='centertop',border_width=5)],
           [sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='Returnpi',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homesimupi',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Salir',button_color='red',size=(3,2),border_width=5,key='Exit1')]],key='pide'),
            sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='Returnpiga',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homesimupiga',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Salir',button_color='red',size=(3,2),border_width=5,key='Exit11')]],key='gapi'),
            sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='Returnpipso',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homesimupipso',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Salir',button_color='red',size=(3,2),border_width=5,key='Exit2')]],key='psooppi')
            ]]

layoutanpd=[[sg.Canvas(key='cananipd')]]
layoutgrapd=[[sg.Canvas(key='cangrapd')]]

layouttappd=[[sg.TabGroup([[sg.Tab('Animación',layoutanpd),sg.Tab('Gráficas', layoutgrapd)]],tab_location='centertop',border_width=5)],
           [sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='Returnpd',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homesimupd',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Salir',button_color='red',size=(3,2),border_width=5,key='Exit2')]],key='tappdde'),
           sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='Returnpdga',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homesimupdga',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Salir',button_color='red',size=(3,2),border_width=5,key='Exit21')]],key='tappdga'),
           sg.Column([[sg.Button(image_filename=r'./img/b1.png', key='Returnpdpso',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='./img/home.png', key='Homesimupdpso',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Salir',button_color='red',size=(3,2),border_width=5,key='Exit22')]],key='tappdpso'),
           ]]

layout1=[[sg.Column(layouthome,key='Home'),sg.Column(layouts, visible=False,key='Sim'),sg.Column(layoutde,key='depara',visible=False),sg.Column(layoutpfps,key='pfps',visible=False),sg.Column(layouttap,key='resps',visible=False),
          sg.Column(layoutga,key='gapara',visible=False),sg.Column(layoutpso,key='psopara',visible=False),
          sg.Column(layouti,key='Inve',visible=False),sg.Column(layoutdepi,key='deparapi',visible=False),sg.Column(layoutpfpi,key='pfpi',visible=False),sg.Column(layouttappi,key='respi',visible=False),
          sg.Column(layoutd,key='Dob',visible=False),sg.Column(layoutdepd,key='deparapd',visible=False),sg.Column(layoutpfpd,key='pfpd',visible=False),sg.Column(layouttappd,key='respd',visible=False)]]
        

window = sg.Window('TT2', layout1, finalize=True,resizable=True)
#Associate fig with Canvas.
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
fig_agg = draw_figure(window['can'].TKCanvas, fig)

figan = plt.figure(figsize=(7, 6))
ax1 = figan.add_subplot(111, autoscale_on=False,xlim=(-1.8, 1.8), ylim=(-1.2, 1.2))
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
fig_anima = draw_figure(window['canani'].TKCanvas, figan)

figgraps = plt.figure(figsize=(7, 6))
ax2 = figgraps.add_subplot(221)
ax3 = figgraps.add_subplot(222)
ax4 = figgraps.add_subplot(223)
fig_graps = draw_figure(window['cangraps'].TKCanvas, figgraps)

figpi = plt.figure(figsize=(6, 5))
ax6 = figpi.add_subplot(111)
fig_aggpi = draw_figure(window['canpfpi'].TKCanvas, figpi)

figanpi = plt.figure(figsize=(8, 6))
ax10 = figanpi.add_subplot(111, autoscale_on=False,xlim=(-1.8, 1.8), ylim=(-1.2, 1.2))
fig_animapi = draw_figure(window['cananipi'].TKCanvas, figanpi)

figgrapspi = plt.figure(figsize=(8, 6))
ax11 = figgrapspi.add_subplot(321)
ax12 = figgrapspi.add_subplot(322)
ax13 = figgrapspi.add_subplot(323)
ax14 = figgrapspi.add_subplot(324)
ax15 = figgrapspi.add_subplot(325)
ax16 = figgrapspi.add_subplot(326)
fig_grapspi = draw_figure(window['cangrapi'].TKCanvas, figgrapspi)

figpd = plt.figure(figsize=(6, 5))
ax30 = figpd.add_subplot(111) 
fig_aggpd = draw_figure(window['canpfpd'].TKCanvas, figpd)
figgrapspd = plt.figure(figsize=(7, 6))
ax21 = figgrapspd.add_subplot(321)
ax22 = figgrapspd.add_subplot(322)
ax23 = figgrapspd.add_subplot(323)
ax24 = figgrapspd.add_subplot(324)
ax25 = figgrapspd.add_subplot(325)
ax26 = figgrapspd.add_subplot(326)
fig_grapd = draw_figure(window['cangrapd'].TKCanvas, figgrapspd)

figanpd = plt.figure(figsize=(7, 6))
ax20 = figanpd.add_subplot(111, autoscale_on=False,xlim=(-2.8,2.8),ylim=(-2.2,2.2))
fig_animapd = draw_figure(window['cananipd'].TKCanvas, figanpd)

layout = 1  # The currently visible layout
while True:
    event, values = window.read()
    print(event, values)
  

    
    #u=float(p)
    #e=eval(values[0])
    if event in (None, 'Exit','Exit0','Exit1','Exit2','Exit10','Exit20','Exit11','Exit12','Exit21','Exit22'):
        break
    if event == 'Simple':
        
        window['Home'].update(visible=False)
        window['Sim'].update(visible=True)
    if event == 'deps':
        window['Sim'].update(visible=False)
        window['depara'].update(visible=True)
        
        ms=values['masaps']
        ls=values['lps']
        lcs=values['lcps']
        bs=values['bps']
        iss=values['ips']
        ss=values['sps']
        try: 
            dinps=np.asarray([ms,ls,lcs,bs,iss,ss], dtype=np.float64, order='C')

            
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['depara'].update(visible=False)
            window['Sim'].update(visible=True)
            
    elif event =='con':
        window['depara'].update(visible=False)
        
    
        window['pfps'].update(visible=True)
        window['Simups'].update(visible=True)
        window['Simupsga'].update(visible=False)
        window['Simupspso'].update(visible=False)
        
        try: 
        
            poblacion=int(values['popb'])
            generaciones=int(values['gen'])
            AMAX=int(values['Am'])
            
            #llamado de la función main de DE
            sg.popup('Ejecución de Evolución Diferencial, espere para poder observar el resultado (conjunto de ganancias para el controlador PID). Las ganancias permitirán al péndulo llegar de la posición inicial a la deseada. Presione ok para continuar con la ejecución')
            var=main(pendulum_s, limit, poblacion, f_mut, recombination, generaciones,dinps,D,M,AMAX)
            valu=np.zeros((len(var[0]),5))
        
            t=var[0]
            s=var[1]
        
            valu[:,0]=s[:,0]
            valu[:,1]=s[:,1]
            valu[:,2]=s[:,2]
            valu[:,3]=t[:,0]
            valu[:,4]=t[:,1]
            indexso=np.argsort(t[:,0])
            valu=valu[indexso]
            
            filename="afa.csv" 
            myFile=open(filename,'w') 
            myFile.write("kp,kd,ki,f1, f2 \n") 
            for l in range(len(t)): 
                myFile.write(str(s[l, 0])+","+str(s[l, 1])+","+str(s[l, 2])+","+str(t[l, 0])+","+str(t[l, 1])+"\n") 
            myFile.close()
        
            #Create a fig for embedding.
            ax.cla()
            ax.set_title('Aproximación al Frente de Pareto')
            ax.set_xlabel('ISE')
            ax.set_ylabel('IADU')
         
            #plot
            ax.scatter(t[:,0], t[:,1])   
            window['Tabl'].update(values=valu)
            #After making changes, fig_agg.draw()Reflect the change with.
            fig_agg.draw()
            nair_scatter = ax.scatter(valu[:, 3], valu[:, 4], c="blue", s=3)
            crs1 = mplcursors.cursor(nair_scatter, hover=True)
            crs1.connect("add", cursor1_annotations)

        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['pfps'].update(visible=False)
            window['depara'].update(visible=True)
    elif event =='rep':
        window['depara'].update(visible=False)
        window['Sim'].update(visible=True)
    elif event =='Homede':
        window['depara'].update(visible=False)
        window['Home'].update(visible=True)

    elif event=='Homeps':
        window['Sim'].update(visible=False)
        window['Home'].update(visible=True)
        

    elif event=='Simups':
        window['pfps'].update(visible=False)
        window['resps'].update(visible=True)
        window['paps'].update(visible=True)
        window['gapaps'].update(visible=False)
        window['psopaps'].update(visible=False)
        
        
        afe=values['Tabl']
        #print(s[afe[0],:])
        pen=pendulum_s(valu[afe[0],:], dinps)
        posi=pen[2]
        tor=pen[3]
        tim=pen[4]
        
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Posición del péndulo [rad]')
        ax2.plot(tim, posi[:, 0], 'k',label=r'$\theta$',lw=1)
        ax2.legend()
        
        
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Velocidad del péndulo [rad/s]')
        ax3.plot(tim, posi[:, 1], 'b',label=r'$\dot{\theta}$',lw=1)
        ax3.legend()
        
        
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Señal de control [Nm]')
        ax4.plot(tim, tor[:, 0], 'r',label=r'$u$',lw=1)
        ax4.legend()
        
        fig_graps.draw()
    
        x0 = np.zeros(len(tim))
        y0 = np.zeros(len(tim))
        xl=np.linspace(-1.8,1.8,len(tim))
        yl=np.linspace(-1.2,1.2,len(tim))
        
        l=dinps[1]
     
        x1 = l * np.sin(posi[:, 0])
        y1 = -l * np.cos(posi[:, 0])
        ax1.cla()
        line, = ax1.plot([], [], 'o-', color='orange', lw=4, \
                markersize=6, markeredgecolor='k', \
                markerfacecolor='k')
        
        time_template = 't= %.1f s'
        time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)
        def init():
            
            line.set_data([], [])
            
            time_text.set_text('')
            return line, time_text
        def animate(i):
            line.set_data([x0[i], x1[i]], [y0[i], y1[i]])
           
            time_text.set_text(time_template % tim[i])
            return line, time_text,
        
        ax1.plot(xl,y0,'k')
        ax1.plot(x0,yl,'k')
        ani_a = animation.FuncAnimation(figan, animate, \
                                np.arange(1, len(tim)), \
                                interval=40, blit=False)
        ani_a.new_frame_seq()
            
        

        
    elif event=='Returnps':
        window['resps'].update(visible=False)
        window['pfps'].update(visible=True)
        
        
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ani_a.event_source.stop()
        ani_a.new_frame_seq()
        
 
    elif event=='Homesimups':
        window['resps'].update(visible=False)
        window['Home'].update(visible=True)
        ax.cla()  
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ani_a.new_frame_seq()
        ani_a.event_source.stop()
        window['Tabl'].update(values=tapsv)
    #------Genetico----------------------------------------
    if event == 'geps':
        window['Sim'].update(visible=False)
        window['gapara'].update(visible=True)
        window['gepips'].update(visible=True)
        window['gepi'].update(visible=False)
        window['gepd'].update(visible=False)
        
        ms=values['masaps']
        ls=values['lps']
        lcs=values['lcps']
        bs=values['bps']
        iss=values['ips']
        ss=values['sps']
        try: 
            dinps=np.asarray([ms,ls,lcs,bs,iss,ss], dtype=np.float64, order='C')

            
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['gapara'].update(visible=False)
            window['Sim'].update(visible=True)
    
    elif event =='repgaps':
        window['gapara'].update(visible=False)
        window['Sim'].update(visible=True)
    elif event =='Homegaps':
        window['gapara'].update(visible=False)
        window['Home'].update(visible=True)
        
    elif event =='congaps':
        window['gapara'].update(visible=False)
        window['Simups'].update(visible=False)
        window['Simupspso'].update(visible=False)
        window['Simupsga'].update(visible=True)
        window['pfps'].update(visible=True)
        #window['Simupd'].update(visible=False)
        
        try: 
        
            pop=int(values['popga'])
            gen=int(values['genga'])
            AMAX=int(values['Amga'])
            eta=int(values['eta'])
                    
                #llamado de la función main de DE
            sg.popup('Ejecución de Algoritmo genético, espere para poder observar el resultado (conjunto de ganancias para el controlador PID). Las ganancias permitirán al péndulo llegar de la posición inicial a la deseada. Presione ok para continuar con la ejecución')
            var= moga( limit, pop,eta, gen,D,M,AMAX,pendulum_s,dinps)
            valu=np.zeros((len(var[0]),5))
                
            t=var[0]
            s=var[1]
                
            valu[:,0]=s[:,0]
            valu[:,1]=s[:,1]
            valu[:,2]=s[:,2]
            valu[:,3]=t[:,0]
            valu[:,4]=t[:,1]
            indexso=np.argsort(t[:,0])
            valu=valu[indexso]
                    
            filename="afaga.csv" 
            myFile=open(filename,'w') 
            myFile.write("kp,kd,ki,f1, f2 \n") 
            for l in range(len(t)): 
                myFile.write(str(s[l, 0])+","+str(s[l, 1])+","+str(s[l, 2])+","+str(t[l, 0])+","+str(t[l, 1])+"\n") 
            myFile.close()
                
                #Create a fig for embedding.
            ax.cla()
            ax.set_title('Aproximación al Frente de Pareto')
            ax.set_xlabel('ISE')
            ax.set_ylabel('IADU')
                
                #plot
            ax.scatter(t[:,0], t[:,1]) 
            nair_scatter = ax.scatter(t[:,0], t[:,1], c="blue", s=3)
            crs1 = mplcursors.cursor(nair_scatter, hover=True)
            crs1.connect("add", cursor1_annotations)  
            window['Tabl'].update(values=valu)
        
        #After making changes, fig_agg.draw()Reflect the change with.
            fig_agg.draw()
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['pfps'].update(visible=False)
            window['gapara'].update(visible=True)
    
    elif event=='Simupsga':
        window['pfps'].update(visible=False)
        
        window['resps'].update(visible=True)
        window['paps'].update(visible=False)
        window['gapaps'].update(visible=True)
        window['psopaps'].update(visible=False)
        
        
        afe=values['Tabl']
        #print(s[afe[0],:])
        pen=pendulum_s(valu[afe[0],:], dinps)
        posi=pen[2]
        tor=pen[3]
        tim=pen[4]
        
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Posición del péndulo [rad]')
        ax2.plot(tim, posi[:, 0], 'k',label=r'$\theta$',lw=1)
        ax2.legend()
        
        
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Velocidad del péndulo [rad/s]')
        ax3.plot(tim, posi[:, 1], 'b',label=r'$\dot{\theta}$',lw=1)
        ax3.legend()
        
        
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Señal de control [Nm]')
        ax4.plot(tim, tor[:, 0], 'r',label=r'$u$',lw=1)
        ax4.legend()
        
        fig_graps.draw()
    
        x0 = np.zeros(len(tim))
        y0 = np.zeros(len(tim))
        xl=np.linspace(-1.8,1.8,len(tim))
        yl=np.linspace(-1.2,1.2,len(tim))
        
        l=dinps[1]
     
        x1 = l * np.sin(posi[:, 0])
        y1 = -l * np.cos(posi[:, 0])
        ax1.cla()
        line, = ax1.plot([], [], 'o-', color='orange', lw=4, \
                markersize=6, markeredgecolor='k', \
                markerfacecolor='k')
        
        time_template = 't= %.1f s'
        time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)
        def init():
            
            line.set_data([], [])
            
            time_text.set_text('')
            return line, time_text
        def animate(i):
            line.set_data([x0[i], x1[i]], [y0[i], y1[i]])
           
            time_text.set_text(time_template % tim[i])
            return line, time_text,
        
        ax1.plot(xl,y0,'k')
        ax1.plot(x0,yl,'k')
        ani_a = animation.FuncAnimation(figan, animate, \
                                np.arange(1, len(tim)), \
                                interval=40, blit=False)
        ani_a.new_frame_seq()
    
    elif event=='Returnpsga':
        window['resps'].update(visible=False)
        window['pfps'].update(visible=True)
        
        
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ani_a.event_source.stop()
        
        
 
    elif event=='Homesimupsga':
        window['resps'].update(visible=False)
        window['Home'].update(visible=True)
        ax.cla()  
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ani_a.new_frame_seq() 
        ani_a.event_source.stop()
        window['Tabl'].update(values=tapsv)
    
    
    #------------------PSO----------------------------------------------------------------------------------------

    if event =='psops':
        window['Sim'].update(visible=False)
        window['psopara'].update(visible=True)
        window['papsops'].update(visible=True)
        window['papsopi'].update(visible=False)
        window['papsopd'].update(visible=False)
        
        
        ms=values['masaps']
        ls=values['lps']
        lcs=values['lcps']
        bs=values['bps']
        iss=values['ips']
        ss=values['sps']
        try: 
            dinps=np.asarray([ms,ls,lcs,bs,iss,ss], dtype=np.float64, order='C')

            
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['psopara'].update(visible=False)
            window['Sim'].update(visible=True)
            
    elif event =='conpsops':
        window['psopara'].update(visible=False)
        
    
        window['pfps'].update(visible=True)
        window['Simups'].update(visible=False)
        window['Simupsga'].update(visible=False)
        window['Simupspso'].update(visible=True)
        Vmin=float(values['Vmin'])
        Vmax=float(values['Vmax'])
        try:
             
            poblacion=int(values['poppso'])
            generaciones=int(values['genpso'])
            AMAX=int(values['Ampso'])
            Vmin=float(values['Vmin'])
            Vmax=float(values['Vmax'])
            c1=int(values['c1'])
            c2=int(values['c2'])

                
            #llamado de la función main de PSO
            sg.popup('Ejecución de PSO, espere para poder observar el resultado (conjunto de ganancias para el controlador PID). Las ganancias permitirán al péndulo llegar de la posición inicial a la deseada. Presione ok para continuar con la ejecución')
            var = MOPSO(pendulum_s, limit, poblacion, Vmax, Vmin, c1, c2, generaciones, dinps, D, M, AMAX)
                
            valu=np.zeros((len(var[0]),5))
            
            t=var[0]
            s=var[1]
            
            valu[:,0]=s[:,0]
            valu[:,1]=s[:,1]
            valu[:,2]=s[:,2]
            valu[:,3]=t[:,0]
            valu[:,4]=t[:,1]
            indexso=np.argsort(t[:,0])
            valu=valu[indexso]
                
            filename="afapso.csv" 
            myFile=open(filename,'w') 
            myFile.write("kp,kd,ki,f1, f2 \n") 
            for l in range(len(t)): 
                myFile.write(str(s[l, 0])+","+str(s[l, 1])+","+str(s[l, 2])+","+str(t[l, 0])+","+str(t[l, 1])+"\n") 
            myFile.close()
            window['Tabl'].update(values=valu)
            #Create a fig for embedding.
            ax.cla()
            ax.set_title('Aproximación al Frente de Pareto')
            ax.set_xlabel('ISE')
            ax.set_ylabel('IADU')
            
            #plot
            ax.scatter(t[:,0], t[:,1])   
            #After making changes, fig_agg.draw()Reflect the change with.
            nair_scatter = ax.scatter(t[:,0], t[:,1], c="blue", s=3)
            crs1 = mplcursors.cursor(nair_scatter, hover=True)
            crs1.connect("add", cursor1_annotations)  
            fig_agg.draw()

        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['pfps'].update(visible=False)
            window['psopara'].update(visible=True)
    elif event =='reppsops':
        window['psopara'].update(visible=False)
        window['Sim'].update(visible=True)
    elif event =='Homepsops':
        window['psopara'].update(visible=False)
        window['Home'].update(visible=True)
    
    elif event=='Simupspso':
        window['pfps'].update(visible=False)
        
        window['resps'].update(visible=True)
        window['paps'].update(visible=False)
        window['gapaps'].update(visible=False)
        window['psopaps'].update(visible=True)
        
        
        
        afe=values['Tabl']
        #print(s[afe[0],:])
        pen=pendulum_s(valu[afe[0],:], dinps)
        posi=pen[2]
        tor=pen[3]
        tim=pen[4]
        
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Posición del péndulo [rad]')
        ax2.plot(tim, posi[:, 0], 'k',label=r'$\theta$',lw=1)
        ax2.legend()
        
        
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Velocidad del péndulo [rad/s]')
        ax3.plot(tim, posi[:, 1], 'b',label=r'$\dot{\theta}$',lw=1)
        ax3.legend()
        
        
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Señal de control [Nm]')
        ax4.plot(tim, tor[:, 0], 'r',label=r'$u$',lw=1)
        ax4.legend()
        
        fig_graps.draw()
    
        x0 = np.zeros(len(tim))
        y0 = np.zeros(len(tim))
        xl=np.linspace(-1.8,1.8,len(tim))
        yl=np.linspace(-1.2,1.2,len(tim))
        
        l=dinps[1]
     
        x1 = l * np.sin(posi[:, 0])
        y1 = -l * np.cos(posi[:, 0])
        ax1.cla()
        line, = ax1.plot([], [], 'o-', color='orange', lw=4, \
                markersize=6, markeredgecolor='k', \
                markerfacecolor='k')
        
        time_template = 't= %.1f s'
        time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)
        def init():
            
            line.set_data([], [])
            
            time_text.set_text('')
            return line, time_text
        def animate(i):
            line.set_data([x0[i], x1[i]], [y0[i], y1[i]])
           
            time_text.set_text(time_template % tim[i])
            return line, time_text,
        
        ax1.plot(xl,y0,'k')
        ax1.plot(x0,yl,'k')
        ani_a = animation.FuncAnimation(figan, animate, \
                                np.arange(1, len(tim)), \
                                interval=40, blit=False)
        ani_a.new_frame_seq()
    
    elif event=='Returnpspso':
        window['resps'].update(visible=False)
        window['pfps'].update(visible=True)
        
        
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ani_a.event_source.stop()
        
        
 
    elif event=='Homesimupspso':
        window['resps'].update(visible=False)
        window['Home'].update(visible=True)
        ax.cla()  
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ani_a.new_frame_seq() 
        ani_a.event_source.stop()
        window['Tabl'].update(values=tapsv)
    

        


    #-------------------------------------------------------------------------------------------------------------
         
    #---------------Invertido-------------------------
    elif event=='Invertido':
        window['Home'].update(visible=False)
        window['Inve'].update(visible=True)
        
    elif event == 'depi':
        window['Inve'].update(visible=False)
        window['deparapi'].update(visible=True)
        
        mi=values['masapi']
        mc=values['masaca']
        li=values['lpi']
        lci=values['lcpi']
        bi=values['bpi']
        bc=values['bca']
        isi=values['ipi']
        sti=values['spi']
        stc=values['spc']
        
        try:
            
            dinpi=np.asarray([mi,mc,li,lci,bi,bc,isi,sti,stc], dtype=np.float64, order='C')
            
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['deparapi'].update(visible=False)
            window['Inve'].update(visible=True)
    elif event =='reppi':
        window['deparapi'].update(visible=False)
        window['Inve'].update(visible=True)
    elif event =='Homedepi':
        window['deparapi'].update(visible=False)
        window['Home'].update(visible=True)
    elif event=='conpi':
        window['deparapi'].update(visible=False)
        window['pfpi'].update(visible=True)
        window['Simupi'].update(visible=True)
        window['Simupiga'].update(visible=False)
        window['Simupipso'].update(visible=False)
        try:
            
            poblacionpi=int(values['popbpi'])
            generacionespi=int(values['genpi'])
            AMAXpi=int(values['Ampi'])
            
            
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['pfpi'].update(visible=False)
            window['deparapi'].update(visible=True)
        try:
            
            
            sg.popup('Ejecución de Evolución Diferencial, espere para poder observar el resultado (conjunto de ganancias para el controlador PID).Las ganancias permitirán al péndulo llegar de la posición inicial a la deseada. Presione ok para continuar con la ejecución')
       
            #llamado de la función main de DE
            varpi=main(inverted_pendulum, limitpi, poblacionpi, f_mutpi, recombinationpi, generacionespi,dinpi,Dpi,Mpi,AMAXpi)
            valupi=np.zeros((len(varpi[0]),(Dpi+Mpi)))

            tpi=varpi[0]
            spi=varpi[1]
        
            valupi[:,0]=spi[:,0]
            valupi[:,1]=spi[:,1]
            valupi[:,2]=spi[:,2]
            valupi[:,3]=spi[:,3]
            valupi[:,4]=spi[:,4]
            valupi[:,5]=spi[:,5]
            valupi[:,6]=tpi[:,0]
            valupi[:,7]=tpi[:,1]
            
            indexsopi=np.argsort(tpi[:,0])
            valupi=valupi[indexsopi]
            
            filename="pifa.csv" 
            myFile=open(filename,'w') 
            myFile.write("kp,kd,ki,kp1,kd1,ki1,f1, f2 \n") 
            for l in range(len(tpi)): 
                myFile.write(str(spi[l, 0])+","+str(spi[l, 1])+","+str(spi[l, 2])+","+str(spi[l, 3])+","+str(spi[l, 4])+","+str(spi[l, 5])+","+str(tpi[l, 0])+","+str(tpi[l, 1])+"\n") 
            myFile.close()

            ax6.set_title('Aproximación al Frente de Pareto')
            ax6.set_xlabel('ISE')
            ax6.set_ylabel('IADU')
        
             
            window['Tablpi'].update(values=valupi)
            #After making changes, fig_agg.draw()Reflect the change with.
            ax6.scatter(tpi[:,0], tpi[:,1])
            nair_scatter = ax6.scatter(tpi[:,0], tpi[:,1], c="blue", s=3)
            crs1 = mplcursors.cursor(nair_scatter, hover=True)
            crs1.connect("add", cursor1_annotations) 
            fig_aggpi.draw()
            
        except:
            sg.popup('Algo salio mal , presione ok e intente de nuevo')
            window['pfpi'].update(visible=False)
            window['deparapi'].update(visible=True)
        
    elif event=='Homepi':
        window['Inve'].update(visible=False)
        window['Home'].update(visible=True)

    elif event=='Simupi':
        window['pfpi'].update(visible=False)
        window['respi'].update(visible=True)
        window['pide'].update(visible=True)
        window['gapi'].update(visible=False)
        window['psooppi'].update(visible=False)
        afepi=values['Tablpi']
       
        penpi=inverted_pendulum(valupi[afepi[0],:], dinpi)
        
        posipi=penpi[2]
        torpi=penpi[3]
        timpi=penpi[4]
        
        ax10.cla()
        ax10.set_xlabel('x [m]')
        ax10.set_ylabel('y [m]')
        
        
        ax11.set_xlabel('Tiempo [s]')
        ax11.set_ylabel('Posición del carro [m]')
        ax11.plot(timpi, posipi[:, 0], 'k',label=r'$x$',lw=1)
        ax11.legend()

        ax12.set_xlabel('Tiempo [s]')
        ax12.set_ylabel('Posición del péndulo [m]')
        ax12.plot(timpi, posipi[:, 1], 'b',label=r'$\theta$',lw=1)
        ax12.legend()
        
        
        ax13.set_xlabel('Tiempo [s]')
        ax13.set_ylabel('Velocidad del carrito [m/s]')
        ax13.plot(timpi, posipi[:, 2], 'r',label=r'$\dot{x}$',lw=1)
        ax13.legend()

        ax14.set_xlabel('Tiempo [s]')
        ax14.set_ylabel('Velocidad del péndulo [m/s]')
        ax14.plot(timpi, posipi[:, 3], 'k',label=r'$\dot{\theta}$',lw=1)
        ax14.legend()
        
    
        ax15.set_xlabel('Tiempo [s]')
        ax15.set_ylabel('$u_{car}$ [Nm]')
        ax15.plot(timpi, torpi[0, :], 'b',label=r'$u_{car}$',lw=1)
        ax15.legend()
        
        
        ax16.set_xlabel('Tiempo [s]')
        ax16.set_ylabel('$u_{pendulum}$ [Nm]')
        ax16.plot(timpi, torpi[1, :], 'r',label=r'$u_{pendulum}$',lw=1)
        ax16.legend()
        
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
        fig_grapspi.draw()

        x0 = np.zeros(len(timpi))
        y0 = np.zeros(len(timpi))
        x1 = posipi[:, 0]
        y1 = np.zeros(len(timpi))
        xlpi=np.linspace(-1.8,1.8,len(timpi))
        ylpi=np.linspace(-1.2,1.2,len(timpi))
        l=dinpi[2]
     
        x2 = l * np.cos(posipi[:, 1]) + x1
        y2 = l * np.sin(posipi[:, 1])
        ax10.cla()
        mass1, = ax10.plot([], [], linestyle='None', marker='s', \
                 markersize=10, markeredgecolor='k', \
                 color='green', markeredgewidth=2)
        line, = ax10.plot([], [], 'o-', color='green', lw=4, \
                markersize=6, markeredgecolor='k', \
                markerfacecolor='k')
        time_template = 't= %.1f s'
        time_text = ax10.text(0.05, 0.9, '', transform=ax10.transAxes)
        def init():
            
            mass1.set_data([], [])
            line.set_data([], [])
            time_text.set_text('')

            return line, mass1, time_text
        
        def animatepi(i):
            mass1.set_data([x1[i]], [y1[i]])
            line.set_data([x1[i], x2[i]], [y1[i], y2[i]])
            time_text.set_text(time_template % timpi[i])
            return mass1, line, time_text
        
        ax10.plot(xlpi,y0,'k')
        ax10.plot(x0,ylpi,'k')
        ani_api = animation.FuncAnimation(figanpi, animatepi, \
                                np.arange(1, len(timpi)), \
                                interval=40, blit=False)
            
    elif event=='Returnpi':
        window['respi'].update(visible=False)
        window['pfpi'].update(visible=True)
        
        ax11.cla()
        ax12.cla()
        ax13.cla()
        ax14.cla()
        ax15.cla()
        ax16.cla()
        ani_api.event_source.stop()
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    elif event=='Homesimupi':
        window['respi'].update(visible=False)
        window['Home'].update(visible=True)
        ax6.cla()
        ax11.cla()
        ax12.cla()
        ax13.cla()
        ax14.cla()
        ax15.cla()
        ax16.cla()
        ani_api.new_frame_seq() 
        ani_api.event_source.stop()
        window['Tablpi'].update(values=tapiv)
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
        
    #-------genético-----------------------------------------------------------
    
    elif event == 'gepipi':
        window['Inve'].update(visible=False)
        window['gapara'].update(visible=True)
        window['gepips'].update(visible=False)
        window['gepi'].update(visible=True)
        window['gepd'].update(visible=False)
        
        mi=values['masapi']
        mc=values['masaca']
        li=values['lpi']
        lci=values['lcpi']
        bi=values['bpi']
        bc=values['bca']
        isi=values['ipi']
        sti=values['spi']
        stc=values['spc']
        
        try:
            
            dinpi=np.asarray([mi,mc,li,lci,bi,bc,isi,sti,stc], dtype=np.float64, order='C')
            
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['gapara'].update(visible=False)
            window['Inve'].update(visible=True)
    
    elif event =='repgapi':
        window['gapara'].update(visible=False)
        window['Inve'].update(visible=True)
    elif event =='Homegapi':
        window['gapara'].update(visible=False)
        window['Home'].update(visible=True)
    elif event=='congapi':
        window['gapara'].update(visible=False)
        window['pfpi'].update(visible=True)
        window['Simupi'].update(visible=False)
        window['Simupiga'].update(visible=True)
        window['Simupipso'].update(visible=False)
        try:
            
            poblacionpi=int(values['popga'])
            generacionespi=int(values['genga'])
            AMAXpi=int(values['Amga'])
            eta=int(values['eta'])
            
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['pfpi'].update(visible=False)
            window['gapara'].update(visible=True)
        try:
            
            
            sg.popup('Ejecución de Algoritmo genético, espere para poder observar el resultado (conjunto de ganancias para el controlador PID).Las ganancias permitirán al péndulo llegar de la posición inicial a la deseada. Presione ok para continuar con la ejecución')
           
            #llamado de la función main de DE
            varpi=moga(limitpi, poblacionpi,eta, generacionespi,Dpi,Mpi,AMAXpi,inverted_pendulum,dinpi)
            valupi=np.zeros((len(varpi[0]),(Dpi+Mpi)))
    
            tpi=varpi[0]
            spi=varpi[1]
            
            valupi[:,0]=spi[:,0]
            valupi[:,1]=spi[:,1]
            valupi[:,2]=spi[:,2]
            valupi[:,3]=spi[:,3]
            valupi[:,4]=spi[:,4]
            valupi[:,5]=spi[:,5]
            valupi[:,6]=tpi[:,0]
            valupi[:,7]=tpi[:,1]
                
            indexsopi=np.argsort(tpi[:,0])
            valupi=valupi[indexsopi]
                
            filename="pifaga.csv" 
            myFile=open(filename,'w') 
            myFile.write("kp,kd,ki,kp1,kd1,ki1,f1, f2 \n") 
            for l in range(len(tpi)): 
                myFile.write(str(spi[l, 0])+","+str(spi[l, 1])+","+str(spi[l, 2])+","+str(spi[l, 3])+","+str(spi[l, 4])+","+str(spi[l, 5])+","+str(tpi[l, 0])+","+str(tpi[l, 1])+"\n") 
            myFile.close()
    
            ax6.set_title('Aproximación al Frente de Pareto')
            ax6.set_xlabel('ISE')
            ax6.set_ylabel('IADU')
            
            ax6.scatter(tpi[:,0], tpi[:,1])
            nair_scatter = ax6.scatter(tpi[:,0], tpi[:,1], c="blue", s=3)
            crs1 = mplcursors.cursor(nair_scatter, hover=True)
            crs1.connect("add", cursor1_annotations)  
            
            window['Tablpi'].update(values=valupi)
            #After making changes, fig_agg.draw()Reflect the change with.
            fig_aggpi.draw()
            
        except:
            sg.popup('Algo salio mal , presione ok e intente de nuevooooo')
            window['pfpi'].update(visible=False)
            window['deparapi'].update(visible=True)
    elif event=='Simupiga':
        window['pfpi'].update(visible=False)
        window['respi'].update(visible=True)
        window['pide'].update(visible=False)
        window['gapi'].update(visible=True)
        window['psooppi'].update(visible=False)
        
        afepi=values['Tablpi']
       
        penpi=inverted_pendulum(valupi[afepi[0],:], dinpi)
        posipi=penpi[2]
        torpi=penpi[3]
        timpi=penpi[4]
        
        ax10.cla()
        ax10.set_xlabel('x [m]')
        ax10.set_ylabel('y [m]')
        
        
        ax11.set_xlabel('Tiempo [s]')
        ax11.set_ylabel('Posición del carro [m]')
        ax11.plot(timpi, posipi[:, 0], 'k',label=r'$x$',lw=1)
        ax11.legend()

        ax12.set_xlabel('Tiempo [s]')
        ax12.set_ylabel('Posición del péndulo [m]')
        ax12.plot(timpi, posipi[:, 1], 'b',label=r'$\theta$',lw=1)
        ax12.legend()
        
        
        ax13.set_xlabel('Tiempo [s]')
        ax13.set_ylabel('Velocidad del carrito [m/s]')
        ax13.plot(timpi, posipi[:, 2], 'r',label=r'$\dot{x}$',lw=1)
        ax13.legend()

        ax14.set_xlabel('Tiempo [s]')
        ax14.set_ylabel('Velocidad del péndulo [m/s]')
        ax14.plot(timpi, posipi[:, 3], 'k',label=r'$\dot{\theta}$',lw=1)
        ax14.legend()
        
    
        ax15.set_xlabel('Tiempo [s]')
        ax15.set_ylabel('$u_{car}$ [Nm]')
        ax15.plot(timpi, torpi[0, :], 'b',label=r'$u_{car}$',lw=1)
        ax15.legend()
        
        
        ax16.set_xlabel('Tiempo [s]')
        ax16.set_ylabel('$u_{pendulum}$ [Nm]')
        ax16.plot(timpi, torpi[1, :], 'r',label=r'$u_{pendulum}$',lw=1)
        ax16.legend()
        
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
        fig_grapspi.draw()

        x0 = np.zeros(len(timpi))
        y0 = np.zeros(len(timpi))
        x1 = posipi[:, 0]
        y1 = np.zeros(len(timpi))
        xlpi=np.linspace(-1.8,1.8,len(timpi))
        ylpi=np.linspace(-1.2,1.2,len(timpi))
        l=dinpi[2]
     
        x2 = l * np.cos(posipi[:, 1]) + x1
        y2 = l * np.sin(posipi[:, 1])
        ax10.cla()
        mass1, = ax10.plot([], [], linestyle='None', marker='s', \
                 markersize=10, markeredgecolor='k', \
                 color='green', markeredgewidth=2)
        line, = ax10.plot([], [], 'o-', color='green', lw=4, \
                markersize=6, markeredgecolor='k', \
                markerfacecolor='k')
        time_template = 't= %.1f s'
        time_text = ax10.text(0.05, 0.9, '', transform=ax10.transAxes)
        def init():
            
            mass1.set_data([], [])
            line.set_data([], [])
            time_text.set_text('')

            return line, mass1, time_text
        
        def animatepi(i):
            mass1.set_data([x1[i]], [y1[i]])
            line.set_data([x1[i], x2[i]], [y1[i], y2[i]])
            time_text.set_text(time_template % timpi[i])
            return mass1, line, time_text
        
        ax10.plot(xlpi,y0,'k')
        ax10.plot(x0,ylpi,'k')
        ani_api = animation.FuncAnimation(figanpi, animatepi, \
                                np.arange(1, len(timpi)), \
                                interval=40, blit=False)
            
    elif event=='Returnpiga':
        window['respi'].update(visible=False)
        window['pfpi'].update(visible=True)
        
        ax11.cla()
        ax12.cla()
        ax13.cla()
        ax14.cla()
        ax15.cla()
        ax16.cla()
        ani_api.event_source.stop()
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    elif event=='Homesimupiga':
        window['respi'].update(visible=False)
        window['Home'].update(visible=True)
        ax6.cla()
        ax11.cla()
        ax12.cla()
        ax13.cla()
        ax14.cla()
        ax15.cla()
        ax16.cla()
        ani_api.new_frame_seq() 
        ani_api.event_source.stop()
        window['Tablpi'].update(values=tapiv)
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    

    #-------------------PSO--------------------------------------------------------------------------------------

    elif event == 'psopi':
        window['Inve'].update(visible=False)
        window['psopara'].update(visible=True)
        window['papsops'].update(visible=False)
        window['papsopi'].update(visible=True)
        window['papsopd'].update(visible=False)
        mi=values['masapi']
        mc=values['masaca']
        li=values['lpi']
        lci=values['lcpi']
        bi=values['bpi']
        bc=values['bca']
        isi=values['ipi']
        sti=values['spi']
        stc=values['spc']
        
        try:
            
            dinpi=np.asarray([mi,mc,li,lci,bi,bc,isi,sti,stc], dtype=np.float64, order='C')
            
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['psopara'].update(visible=False)
            window['Inve'].update(visible=True)
    elif event =='reppsopi':
        window['psopara'].update(visible=False)
        window['Inve'].update(visible=True)
    elif event =='Homepsopi':
        window['psopara'].update(visible=False)
        window['Home'].update(visible=True)
    elif event=='conpsopi':
        window['psopara'].update(visible=False)
        window['pfpi'].update(visible=True)
        window['Simupi'].update(visible=False)
        window['Simupiga'].update(visible=False)
        window['Simupipso'].update(visible=True)
        try:
            
            poblacionpi=int(values['poppso'])
            generacionespi=int(values['genpso'])
            AMAXpi=int(values['Ampso'])
            Vminpi=float(values['Vmin'])
            Vmaxpi=float(values['Vmax'])
            c1pi=int(values['c1'])
            c2pi=int(values['c2'])
            
            
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['pfpi'].update(visible=False)
            window['psopara'].update(visible=True)
        try:
            
            
            sg.popup('Ejecución de PSO, espere para poder observar el resultado (conjunto de ganancias para el controlador PID).Las ganancias permitirán al péndulo llegar de la posición inicial a la deseada. Presione ok para continuar con la ejecución')
       
            #llamado de la función main de DE
            varpi = MOPSO(inverted_pendulum, limitpi, poblacionpi, Vmaxpi, Vminpi,c1pi, c2pi, generacionespi,dinpi, Dpi, Mpi, AMAXpi)
            
            valupi=np.zeros((len(varpi[0]),(Dpi+Mpi)))

            tpi=varpi[0]
            spi=varpi[1]
        
            valupi[:,0]=spi[:,0]
            valupi[:,1]=spi[:,1]
            valupi[:,2]=spi[:,2]
            valupi[:,3]=spi[:,3]
            valupi[:,4]=spi[:,4]
            valupi[:,5]=spi[:,5]
            valupi[:,6]=tpi[:,0]
            valupi[:,7]=tpi[:,1]
            
            indexsopi=np.argsort(tpi[:,0])
            valupi=valupi[indexsopi]
            
            filename="pifapso.csv" 
            myFile=open(filename,'w') 
            myFile.write("kp,kd,ki,kp1,kd1,ki1,f1, f2 \n") 
            for l in range(len(tpi)): 
                myFile.write(str(spi[l, 0])+","+str(spi[l, 1])+","+str(spi[l, 2])+","+str(spi[l, 3])+","+str(spi[l, 4])+","+str(spi[l, 5])+","+str(tpi[l, 0])+","+str(tpi[l, 1])+"\n") 
            myFile.close()

            ax6.set_title('Aproximación al Frente de Pareto')
            ax6.set_xlabel('ISE')
            ax6.set_ylabel('IADU')
        
            ax6.scatter(tpi[:,0], tpi[:,1])
            nair_scatter = ax6.scatter(tpi[:,0], tpi[:,1], c="blue", s=3)
            crs1 = mplcursors.cursor(nair_scatter, hover=True)
            crs1.connect("add", cursor1_annotations)  
        
            window['Tablpi'].update(values=valupi)
            #After making changes, fig_agg.draw()Reflect the change with.
            fig_aggpi.draw()
            
        except:
            sg.popup('Algo salio mal , presione ok e intente de nuevo')
            window['pfpi'].update(visible=False)
            window['psopara'].update(visible=True)
    
    elif event=='Simupipso':
        window['pfpi'].update(visible=False)
        window['respi'].update(visible=True)
        window['pide'].update(visible=False)
        window['gapi'].update(visible=False)
        window['psooppi'].update(visible=True)
        
        afepi=values['Tablpi']
       
        penpi=inverted_pendulum(valupi[afepi[0],:], dinpi)
        
        posipi=penpi[2]
        torpi=penpi[3]
        timpi=penpi[4]
        
        ax10.cla()
        ax10.set_xlabel('x [m]')
        ax10.set_ylabel('y [m]')
        
        
        ax11.set_xlabel('Tiempo [s]')
        ax11.set_ylabel('Posición del carro [m]')
        ax11.plot(timpi, posipi[:, 0], 'k',label=r'$x$',lw=1)
        ax11.legend()

        ax12.set_xlabel('Tiempo [s]')
        ax12.set_ylabel('Posición del péndulo [m]')
        ax12.plot(timpi, posipi[:, 1], 'b',label=r'$\theta$',lw=1)
        ax12.legend()
        
        
        ax13.set_xlabel('Tiempo [s]')
        ax13.set_ylabel('Velocidad del carrito [m/s]')
        ax13.plot(timpi, posipi[:, 2], 'r',label=r'$\dot{x}$',lw=1)
        ax13.legend()

        ax14.set_xlabel('Tiempo [s]')
        ax14.set_ylabel('Velocidad del péndulo [m/s]')
        ax14.plot(timpi, posipi[:, 3], 'k',label=r'$\dot{\theta}$',lw=1)
        ax14.legend()
        
    
        ax15.set_xlabel('Tiempo [s]')
        ax15.set_ylabel('$u_{car}$ [Nm]')
        ax15.plot(timpi, torpi[0, :], 'b',label=r'$u_{car}$',lw=1)
        ax15.legend()
        
        
        ax16.set_xlabel('Tiempo [s]')
        ax16.set_ylabel('$u_{pendulum}$ [Nm]')
        ax16.plot(timpi, torpi[1, :], 'r',label=r'$u_{pendulum}$',lw=1)
        ax16.legend()
        
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
        fig_grapspi.draw()

        x0 = np.zeros(len(timpi))
        y0 = np.zeros(len(timpi))
        x1 = posipi[:, 0]
        y1 = np.zeros(len(timpi))
        xlpi=np.linspace(-1.8,1.8,len(timpi))
        ylpi=np.linspace(-1.2,1.2,len(timpi))
        l=dinpi[2]
     
        x2 = l * np.cos(posipi[:, 1]) + x1
        y2 = l * np.sin(posipi[:, 1])
        ax10.cla()
        mass1, = ax10.plot([], [], linestyle='None', marker='s', \
                 markersize=10, markeredgecolor='k', \
                 color='green', markeredgewidth=2)
        line, = ax10.plot([], [], 'o-', color='green', lw=4, \
                markersize=6, markeredgecolor='k', \
                markerfacecolor='k')
        time_template = 't= %.1f s'
        time_text = ax10.text(0.05, 0.9, '', transform=ax10.transAxes)
        def init():
            
            mass1.set_data([], [])
            line.set_data([], [])
            time_text.set_text('')

            return line, mass1, time_text
        
        def animatepi(i):
            mass1.set_data([x1[i]], [y1[i]])
            line.set_data([x1[i], x2[i]], [y1[i], y2[i]])
            time_text.set_text(time_template % timpi[i])
            return mass1, line, time_text
        
        ax10.plot(xlpi,y0,'k')
        ax10.plot(x0,ylpi,'k')
        ani_api = animation.FuncAnimation(figanpi, animatepi, \
                                np.arange(1, len(timpi)), \
                                interval=40, blit=False)
            
    elif event=='Returnpipso':
        window['respi'].update(visible=False)
        window['pfpi'].update(visible=True)
        
        ax11.cla()
        ax12.cla()
        ax13.cla()
        ax14.cla()
        ax15.cla()
        ax16.cla()
        ani_api.event_source.stop()
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    elif event=='Homesimupipso':
        window['respi'].update(visible=False)
        window['Home'].update(visible=True)
        ax6.cla()
        ax11.cla()
        ax12.cla()
        ax13.cla()
        ax14.cla()
        ax15.cla()
        ax16.cla()
        ani_api.new_frame_seq() 
        ani_api.event_source.stop()
        window['Tablpi'].update(values=tapiv)
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    



    #------------------------------------------------------------------------------------------------------------
    
    #------Doble-------------------------------------------------------------------------------------------------
        
    elif event == 'Doble':
    
        window['Home'].update(visible=False)
        window['Dob'].update(visible=True)
        
    elif event == 'depd':
        window['Dob'].update(visible=False)
        window['deparapd'].update(visible=True)
         
        m1=values['masapd1']
        m2=values['masapd2']
        l1=values['lpd1']
        lc1=values['lcpd1']
        l2=values['lpd2']
        lc2=values['lcpd2']
        b1=values['bpd1']
        b2=values['bpd2']
        isd1=values['ipd1']
        isd2=values['ipd2']
        
        try:
            dinpd=np.asarray([m1,m2,l1,lc1,l2,lc2,b1,b2,isd1,isd2], dtype=np.float64, order='C')
            
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['deparapd'].update(visible=False)
            window['Dob'].update(visible=True)
    
    elif event =='reppd':
        window['deparapd'].update(visible=False)
        window['Dob'].update(visible=True)
    elif event =='Homedepd':
        window['deparapd'].update(visible=False)
        window['Home'].update(visible=True)       
    elif event=='conpd':
        
        window['deparapd'].update(visible=False)
        window['pfpd'].update(visible=True)
        window['Simupd'].update(visible=True)
        window['Simupdpso'].update(visible=False)
        window['Simupdga'].update(visible=False)

        try:
            
            poblacionpd=int(values['popbpd'])
            generacionespd=int(values['genpd'])
            AMAXpd=int(values['Ampd'])

            
            sg.popup('Ejecución de Evolución Diferencial, espere para poder observar el resultado (conjunto de ganancias para el controlador PID). Las ganancias permitirán al péndulo seguir la trayectoria deseada. Presione ok para continuar con la ejecución')
            varpd=main(double_pendulum, limitpd, poblacionpd, f_mutpd, recombinationpd, generacionespd,dinpd,Dpd,Mpd,AMAXpd)
        
            valupd=np.zeros((len(varpd[0]),(Dpd+Mpd)))

            tpd=varpd[0]
            spd=varpd[1]
        
            valupd[:,0]=spd[:,0]
            valupd[:,1]=spd[:,1]
            valupd[:,2]=spd[:,2]
            valupd[:,3]=spd[:,3]
            valupd[:,4]=tpd[:,0]
            valupd[:,5]=tpd[:,1]
            
            indexsopd=np.argsort(tpd[:,0])
            valupd=valupd[indexsopd]
            
            filename="pdfa.csv" 
            myFile=open(filename,'w') 
            myFile.write("kp,kd,kp1,kd1,f1, f2 \n") 
            for l in range(len(tpd)): 
                myFile.write(str(spd[l, 0])+","+str(spd[l, 1])+","+str(spd[l, 2])+","+str(spd[l, 3])+","+str(tpd[l, 0])+","+str(tpd[l, 1])+"\n") 
            myFile.close()

            #Create a fig for embedding.
            
            ax30.set_title('Aproximación al frente de Pareto')
            ax30.set_xlabel('ISE')
            ax30.set_ylabel('IADU')
        
            #plot
            ax30.scatter(tpd[:,0], tpd[:,1])
            nair_scatter = ax30.scatter(tpd[:,0], tpd[:,1], c="blue", s=3)
            crs1 = mplcursors.cursor(nair_scatter, hover=True)
            crs1.connect("add", cursor1_annotations) 

            window['Tablpd'].update(values=valupd)
            #After making changes, fig_agg.draw()Reflect the change with.
            fig_aggpd.draw()
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['pfpd'].update(visible=False)
            window['deparapd'].update(visible=True)
        
    elif event=='Homepd':
        window['Dob'].update(visible=False)
        window['Home'].update(visible=True)
    elif event=='Simupd':
        window['pfpd'].update(visible=False)
        window['respd'].update(visible=True)
        window['tappdde'].update(visible=True)
        window['tappdga'].update(visible=False)
        window['tappdpso'].update(visible=False)
        
        afepd=values['Tablpd']
       
        penpd=double_pendulum(valupd[afepd[0],:], dinpd)
        posipd=penpd[2]
        torpd=penpd[3]
        timpd=penpd[4]
        
     
        ax21.set_xlabel('Tiempo')
        ax21.set_ylabel('Posición de la barra 1 ')
        ax21.plot(timpd, posipd[:, 0], 'k',label=r'$\theta_1$',lw=1)
        ax21.legend()
        
        
        ax22.set_xlabel('Tiempo')
        ax22.set_ylabel('Posición de la barra 2')
        ax22.plot(timpd, posipd[:, 1], 'b',label=r'$\theta_2$',lw=1)
        ax22.legend()
        

        ax23.set_xlabel('Tiempo')
        ax23.set_ylabel('Velocidad de la barra 1')
        ax23.plot(timpd, posipd[:, 2], 'r',label=r'$\dot{\theta_1}$',lw=1)
        ax23.legend()
        
        
        ax24.set_xlabel('Tiempo')
        ax24.set_ylabel('Velocidad de la barra 2')
        ax24.plot(timpd, posipd[:, 3], 'k',label=r'$\dot{\theta_1}$',lw=1)
        ax24.legend()
        
        
        ax25.set_xlabel('Tiempo')
        ax25.set_ylabel('$u_{1}$')
        ax25.plot(timpd[:2995], torpd[0, :2995], 'b',label=r'$u_{1}$',lw=1)
        ax25.legend()
        
        
        ax26.set_xlabel('Tiempo')
        ax26.set_ylabel('$u_{2}$')
        ax26.plot(timpd[:2995], torpd[1,:2995], 'r',label=r'$u_{2}$',lw=1)
        ax26.legend()
        
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
        
        fig_grapd.draw()
    
        ax20.cla()
        ax20.set_xlabel('x')
        ax20.set_ylabel('y')
        
    
        l1=dinpd[2]
        l2=dinpd[4]
        
        x0=np.zeros(len(timpd))
        y0=np.zeros(len(timpd))
        
        xlpd=np.linspace(-2.8,2.8,len(timpd))
        ylpd=np.linspace(-2.2,2.2,len(timpd))

        x1=l1*np.sin(posipd[:,0])
        y1=-l1*np.cos(posipd[:,0])


        x2=l1*np.sin(posipd[:,0])+ l2*np.sin(posipd[:,0]+posipd[:,1])
        y2=-l1*np.cos(posipd[:,0])- l2*np.cos(posipd[:,0]+posipd[:,1])
   
        #line1, = ax.plot([], [], 'o-',color = 'g',lw=4, markersize = 15, markeredgecolor = 'k',markerfacecolor = 'r',markevery=10000)
        line, = ax20.plot([], [], 'o-',color = 'g',markersize = 3, markerfacecolor = 'k',lw=2, markevery=100000, markeredgecolor = 'k')   # line for Earth
        line1, = ax20.plot([], [], 'o-',color = 'r',markersize = 8, markerfacecolor = 'b',lw=2, markevery=100000, markeredgecolor = 'k')   # line for Jupiter
        line2, = ax20.plot([], [], 'o-',color = 'k',markersize = 8, markerfacecolor = 'r',lw=1, markevery=1000000, markeredgecolor = 'k')  



        time_template = 't= %.1f s'
        time_text = ax20.text(0.05,0.9,'',transform=ax20.transAxes)


        def init():
            line.set_data([],[])
            line1.set_data([],[])
            line2.set_data([], [])
            time_text.set_text('')
            
            return line, time_text, line1,

        def animatepd(i):
            #trail1 =  16
            trail2 = 1100   
            
            line.set_data([x0[i],x1[i]],[y0[i],y1[i]])
            line1.set_data([x1[i],x2[i]],[y1[i],y2[i]])
            line2.set_data(x2[i:max(1,i-trail2):-1], y2[i:max(1,i-trail2):-1])
            time_text.set_text(time_template % timpd[i])
            
            return line, time_text, line1,line2
        ax20.plot(x0,ylpd, 'k',lw=1)
        ax20.plot(xlpd, y0,'k',lw=1)

        ani_apd = animation.FuncAnimation(figanpd, animatepd, \
                 np.arange(1,len(timpd)), \
                 interval=1,blit=False,init_func=init)
        ani_apd.new_frame_seq() 
        
        
    elif event=='Returnpd':
        window['respd'].update(visible=False)
        window['pfpd'].update(visible=True)
        
        ax21.cla()
        ax22.cla()
        ax23.cla()
        ax24.cla()
        ax25.cla()
        ax26.cla()
      
        ani_apd.event_source.stop()
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    elif event=='Homesimupd':
        window['respd'].update(visible=False)
        window['Home'].update(visible=True)
        ax30.cla()
        ax21.cla()
        ax22.cla()
        ax23.cla()
        ax24.cla()
        ax25.cla()
        ax6.cla()
        ani_apd.event_source.stop()
        ani_apd.new_frame_seq() 
        window['Tablpd'].update(values=tapdv)
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)

    #---------------------Genético ------------------------------------------------------------------
    elif event == 'gepd2':
        window['Dob'].update(visible=False)
        window['gapara'].update(visible=True)
        window['gepips'].update(visible=False)
        window['gepi'].update(visible=False)
        window['gepd'].update(visible=True)
        
        m1=values['masapd1']
        m2=values['masapd2']
        l1=values['lpd1']
        lc1=values['lcpd1']
        l2=values['lpd2']
        lc2=values['lcpd2']
        b1=values['bpd1']
        b2=values['bpd2']
        isd1=values['ipd1']
        isd2=values['ipd2']
        
        try:
            dinpd=np.asarray([m1,m2,l1,lc1,l2,lc2,b1,b2,isd1,isd2], dtype=np.float64, order='C')
            
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['deparapd'].update(visible=False)
            window['Dob'].update(visible=True)
    
    elif event =='repgapd':
        window['gapara'].update(visible=False)
        window['Dob'].update(visible=True)
    elif event =='Homegapd':
        window['gapara'].update(visible=False)
        window['Home'].update(visible=True) 
    
    elif event=='congapd':
        
        window['gapara'].update(visible=False)
        window['pfpd'].update(visible=True)
        window['Simupd'].update(visible=False)
        window['Simupdpso'].update(visible=False)
        window['Simupdga'].update(visible=True)
        try:
            
            poblacionpd=int(values['popga'])
            generacionespd=int(values['genga'])
            AMAXpd=int(values['Amga'])
            eta=int(values['eta'])
            sg.popup('Ejecución de Algoritmo genético, espere para poder observar el resultado (conjunto de ganancias para el controlador PID). Las ganancias permitirán al péndulo seguir la trayectoria deseada. Presione ok para continuar con la ejecución')
            
        
            #llamado de la función main de DE
            varpd=moga(limitpd, poblacionpd,eta, generacionespd,Dpd,Mpd,AMAXpd,double_pendulum,dinpd)
        
            valupd=np.zeros((len(varpd[0]),(Dpd+Mpd)))
            tpd=varpd[0]
            spd=varpd[1]
        
            valupd[:,0]=spd[:,0]
            valupd[:,1]=spd[:,1]
            valupd[:,2]=spd[:,2]
            valupd[:,3]=spd[:,3]
            valupd[:,4]=tpd[:,0]
            valupd[:,5]=tpd[:,1]
            
            indexsopd=np.argsort(tpd[:,0])
            valupd=valupd[indexsopd]
            
            filename="pdfaga.csv" 
            myFile=open(filename,'w') 
            myFile.write("kp,kd,kp1,kd1,f1, f2 \n") 
            for l in range(len(tpd)): 
                myFile.write(str(spd[l, 0])+","+str(spd[l, 1])+","+str(spd[l, 2])+","+str(spd[l, 3])+","+str(tpd[l, 0])+","+str(tpd[l, 1])+"\n") 
            myFile.close()

            #Create a fig for embedding.
            
            ax30.set_title('Aproximación al frente de Pareto')
            ax30.set_xlabel('ISE')
            ax30.set_ylabel('IADU')
        
            #plot
            ax30.scatter(tpd[:,0], tpd[:,1])
            nair_scatter = ax30.scatter(tpd[:,0], tpd[:,1], c="blue", s=3)
            crs1 = mplcursors.cursor(nair_scatter, hover=True)
            crs1.connect("add", cursor1_annotations) 

            window['Tablpd'].update(values=valupd)
            #After making changes, fig_agg.draw()Reflect the change with.
            fig_aggpd.draw()
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['pfpd'].update(visible=False)
            window['deparapd'].update(visible=True)
        
    elif event=='Homepd':
        window['Dob'].update(visible=False)
        window['Home'].update(visible=True)
    
    elif event=='Simupdga':
        window['pfpd'].update(visible=False)
        window['respd'].update(visible=True)
        window['tappdde'].update(visible=False)
        window['tappdga'].update(visible=True)
        window['tappdpso'].update(visible=False)
        
        
        afepd=values['Tablpd']
       
        penpd=double_pendulum(valupd[afepd[0],:], dinpd)
        posipd=penpd[2]
        torpd=penpd[3]
        timpd=penpd[4]
        
     
        ax21.set_xlabel('Tiempo')
        ax21.set_ylabel('Posición de la barra 1 ')
        ax21.plot(timpd, posipd[:, 0], 'k',label=r'$\theta_1$',lw=1)
        ax21.legend()
        
        
        ax22.set_xlabel('Tiempo')
        ax22.set_ylabel('Posición de la barra 2')
        ax22.plot(timpd, posipd[:, 1], 'b',label=r'$\theta_2$',lw=1)
        ax22.legend()
        

        ax23.set_xlabel('Tiempo')
        ax23.set_ylabel('Velocidad de la barra 1')
        ax23.plot(timpd, posipd[:, 2], 'r',label=r'$\dot{\theta_1}$',lw=1)
        ax23.legend()
        
        
        ax24.set_xlabel('Tiempo')
        ax24.set_ylabel('Velocidad de la barra 2')
        ax24.plot(timpd, posipd[:, 3], 'k',label=r'$\dot{\theta_1}$',lw=1)
        ax24.legend()
        
        
        ax25.set_xlabel('Tiempo')
        ax25.set_ylabel('$u_{1}$')
        ax25.plot(timpd[:2995], torpd[0, :2995], 'b',label=r'$u_{1}$',lw=1)
        ax25.legend()
        
        
        ax26.set_xlabel('Tiempo')
        ax26.set_ylabel('$u_{2}$')
        ax26.plot(timpd[:2995], torpd[1,:2995], 'r',label=r'$u_{2}$',lw=1)
        ax26.legend()
        
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
        
        fig_grapd.draw()
    
        ax20.cla()
        ax20.set_xlabel('x')
        ax20.set_ylabel('y')
        
    
        l1=dinpd[2]
        l2=dinpd[4]
        
        x0=np.zeros(len(timpd))
        y0=np.zeros(len(timpd))
        
        xlpd=np.linspace(-2.8,2.8,len(timpd))
        ylpd=np.linspace(-2.2,2.2,len(timpd))

        x1=l1*np.sin(posipd[:,0])
        y1=-l1*np.cos(posipd[:,0])

        x2=l1*np.sin(posipd[:,0])+ l2*np.sin(posipd[:,0]+posipd[:,1])
        y2=-l1*np.cos(posipd[:,0])- l2*np.cos(posipd[:,0]+posipd[:,1])
   
        #line1, = ax.plot([], [], 'o-',color = 'g',lw=4, markersize = 15, markeredgecolor = 'k',markerfacecolor = 'r',markevery=10000)
        line, = ax20.plot([], [], 'o-',color = 'g',markersize = 3, markerfacecolor = 'k',lw=2, markevery=100000, markeredgecolor = 'k')   # line for Earth
        line1, = ax20.plot([], [], 'o-',color = 'r',markersize = 8, markerfacecolor = 'b',lw=2, markevery=100000, markeredgecolor = 'k')   # line for Jupiter
        line2, = ax20.plot([], [], 'o-',color = 'k',markersize = 8, markerfacecolor = 'r',lw=1, markevery=1000000, markeredgecolor = 'k')  



        time_template = 't= %.1f s'
        time_text = ax20.text(0.05,0.9,'',transform=ax20.transAxes)


        def init():
            line.set_data([],[])
            line1.set_data([],[])
            line2.set_data([], [])
            time_text.set_text('')
            
            return line, time_text, line1,

        def animatepd(i):
            #trail1 =  16
            trail2 = 1100   
            
            line.set_data([x0[i],x1[i]],[y0[i],y1[i]])
            line1.set_data([x1[i],x2[i]],[y1[i],y2[i]])
            line2.set_data(x2[i:max(1,i-trail2):-1], y2[i:max(1,i-trail2):-1])
            time_text.set_text(time_template % timpd[i])
            
            return line, time_text, line1,line2
        ax20.plot(x0,ylpd, 'k',lw=1)
        ax20.plot(xlpd, y0,'k',lw=1)

        ani_apd = animation.FuncAnimation(figanpd, animatepd, \
                 np.arange(1,len(timpd)), \
                 interval=1,blit=False,init_func=init)
        ani_apd.new_frame_seq() 
        
        
    elif event=='Returnpdga':
        window['respd'].update(visible=False)
        window['pfpd'].update(visible=True)
        
        ax21.cla()
        ax22.cla()
        ax23.cla()
        ax24.cla()
        ax25.cla()
        ax26.cla()
      
        ani_apd.event_source.stop()
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    elif event=='Homesimupdga':
        window['respd'].update(visible=False)
        window['Home'].update(visible=True)
        ax30.cla()
        ax21.cla()
        ax22.cla()
        ax23.cla()
        ax24.cla()
        ax25.cla()
        ax6.cla()
        ani_apd.event_source.stop()
        ani_apd.new_frame_seq() 
        window['Tablpd'].update(values=tapdv)
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    #--------------------------- PSO -------------------------------------------------------------
    elif event == 'psopd':
        window['Dob'].update(visible=False)
        window['psopara'].update(visible=True)
        window['papsops'].update(visible=False)
        window['papsopi'].update(visible=False)
        window['papsopd'].update(visible=True)
        
        m1=values['masapd1']
        m2=values['masapd2']
        l1=values['lpd1']
        lc1=values['lcpd1']
        l2=values['lpd2']
        lc2=values['lcpd2']
        b1=values['bpd1']
        b2=values['bpd2']
        isd1=values['ipd1']
        isd2=values['ipd2']
        
        try:
            dinpd=np.asarray([m1,m2,l1,lc1,l2,lc2,b1,b2,isd1,isd2], dtype=np.float64, order='C')
            
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['psopara'].update(visible=False)
            window['Dob'].update(visible=True)
    
    elif event =='repsopd':
        window['psopara'].update(visible=False)
        window['Dob'].update(visible=True)
    elif event =='Homepsodepd':
        window['psopara'].update(visible=False)
        window['Home'].update(visible=True)       
    elif event=='conpsopd':
        
        window['psopara'].update(visible=False)
        window['pfpd'].update(visible=True)
        window['Simupd'].update(visible=False)
        window['Simupdpso'].update(visible=False)
        window['Simupdga'].update(visible=True)

        try:
            
            poblacionpd=int(values['poppso'])
            generacionespd=int(values['genpso'])
            AMAXpd=int(values['Ampso'])
            Vminpd=float(values['Vmin'])
            Vmaxpd=float(values['Vmax'])
            c1pd=int(values['c1'])
            c2pd=int(values['c2'])
            sg.popup('Ejecución de PSO, espere para poder observar el resultado (conjunto de ganancias para el controlador PID). Las ganancias permitirán al péndulo seguir la trayectoria deseada. Presione ok para continuar con la ejecución')
            
        
            #llamado de la función main de DE

            varpd = MOPSO(double_pendulum, limitpd, poblacionpd, Vmaxpd, Vminpd,c1pd, c2pd, generacionespd,dinpd,Dpd,Mpd,AMAXpd)
            valupd=np.zeros((len(varpd[0]),(Dpd+Mpd)))

            tpd=varpd[0]
            spd=varpd[1]
        
            valupd[:,0]=spd[:,0]
            valupd[:,1]=spd[:,1]
            valupd[:,2]=spd[:,2]
            valupd[:,3]=spd[:,3]
            valupd[:,4]=tpd[:,0]
            valupd[:,5]=tpd[:,1]
            
            indexsopd=np.argsort(tpd[:,0])
            valupd=valupd[indexsopd]
            
            filename="pdfa.csv" 
            myFile=open(filename,'w') 
            myFile.write("kp,kd,kp1,kd1,f1, f2 \n") 
            for l in range(len(tpd)): 
                myFile.write(str(spd[l, 0])+","+str(spd[l, 1])+","+str(spd[l, 2])+","+str(spd[l, 3])+","+str(tpd[l, 0])+","+str(tpd[l, 1])+"\n") 
            myFile.close()

            #Create a fig for embedding.
            
            ax30.set_title('Aproximación al frente de Pareto')
            ax30.set_xlabel('ISE')
            ax30.set_ylabel('IADU')
        
            #plot
            ax30.scatter(tpd[:,0], tpd[:,1])
            nair_scatter = ax30.scatter(tpd[:,0], tpd[:,1], c="blue", s=3)
            crs1 = mplcursors.cursor(nair_scatter, hover=True)
            crs1.connect("add", cursor1_annotations) 

            window['Tablpd'].update(values=valupd)
            #After making changes, fig_agg.draw()Reflect the change with.
            fig_aggpd.draw()
        except:
            sg.popup('Todos los datos ingresados deben ser númericos, presione ok e intente de nuevo')
            window['pfpd'].update(visible=False)
            window['psopara'].update(visible=True)
        
    
    elif event=='Simupdpso':
        window['pfpd'].update(visible=False)
        window['respd'].update(visible=True)
        window['tappdde'].update(visible=False)
        window['tappdga'].update(visible=False)
        window['tappdpso'].update(visible=True)
        
        afepd=values['Tablpd']
       
        penpd=double_pendulum(valupd[afepd[0],:], dinpd)
        posipd=penpd[2]
        torpd=penpd[3]
        timpd=penpd[4]
        
     
        ax21.set_xlabel('Tiempo')
        ax21.set_ylabel('Posición de la barra 1 ')
        ax21.plot(timpd, posipd[:, 0], 'k',label=r'$\theta_1$',lw=1)
        ax21.legend()
        
        
        ax22.set_xlabel('Tiempo')
        ax22.set_ylabel('Posición de la barra 2')
        ax22.plot(timpd, posipd[:, 1], 'b',label=r'$\theta_2$',lw=1)
        ax22.legend()
        

        ax23.set_xlabel('Tiempo')
        ax23.set_ylabel('Velocidad de la barra 1')
        ax23.plot(timpd, posipd[:, 2], 'r',label=r'$\dot{\theta_1}$',lw=1)
        ax23.legend()
        
        
        ax24.set_xlabel('Tiempo')
        ax24.set_ylabel('Velocidad de la barra 2')
        ax24.plot(timpd, posipd[:, 3], 'k',label=r'$\dot{\theta_1}$',lw=1)
        ax24.legend()
        
        
        ax25.set_xlabel('Tiempo')
        ax25.set_ylabel('$u_{1}$')
        ax25.plot(timpd[:2995], torpd[0, :2995], 'b',label=r'$u_{1}$',lw=1)
        ax25.legend()
        
        
        ax26.set_xlabel('Tiempo')
        ax26.set_ylabel('$u_{2}$')
        ax26.plot(timpd[:2995], torpd[1,:2995], 'r',label=r'$u_{2}$',lw=1)
        ax26.legend()
        
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
        
        fig_grapd.draw()
    
        ax20.cla()
        ax20.set_xlabel('x')
        ax20.set_ylabel('y')
        
    
        l1=dinpd[2]
        l2=dinpd[4]
        
        x0=np.zeros(len(timpd))
        y0=np.zeros(len(timpd))
        
        xlpd=np.linspace(-2.8,2.8,len(timpd))
        ylpd=np.linspace(-2.2,2.2,len(timpd))

        x1=l1*np.sin(posipd[:,0])
        y1=-l1*np.cos(posipd[:,0])

        x2=l1*np.sin(posipd[:,0])+ l2*np.sin(posipd[:,0]+posipd[:,1])
        y2=-l1*np.cos(posipd[:,0])- l2*np.cos(posipd[:,0]+posipd[:,1])
   
        #line1, = ax.plot([], [], 'o-',color = 'g',lw=4, markersize = 15, markeredgecolor = 'k',markerfacecolor = 'r',markevery=10000)
        line, = ax20.plot([], [], 'o-',color = 'g',markersize = 3, markerfacecolor = 'k',lw=2, markevery=100000, markeredgecolor = 'k')   # line for Earth
        line1, = ax20.plot([], [], 'o-',color = 'r',markersize = 8, markerfacecolor = 'b',lw=2, markevery=100000, markeredgecolor = 'k')   # line for Jupiter
        line2, = ax20.plot([], [], 'o-',color = 'k',markersize = 8, markerfacecolor = 'r',lw=1, markevery=1000000, markeredgecolor = 'k')  



        time_template = 't= %.1f s'
        time_text = ax20.text(0.05,0.9,'',transform=ax20.transAxes)


        def init():
            line.set_data([],[])
            line1.set_data([],[])
            line2.set_data([], [])
            time_text.set_text('')
            
            return line, time_text, line1,

        def animatepd(i):
            #trail1 =  16
            trail2 = 1100   
            
            line.set_data([x0[i],x1[i]],[y0[i],y1[i]])
            line1.set_data([x1[i],x2[i]],[y1[i],y2[i]])
            line2.set_data(x2[i:max(1,i-trail2):-1], y2[i:max(1,i-trail2):-1])
            time_text.set_text(time_template % timpd[i])
            
            return line, time_text, line1,line2
        ax20.plot(x0,ylpd, 'k',lw=1)
        ax20.plot(xlpd, y0,'k',lw=1)

        ani_apd = animation.FuncAnimation(figanpd, animatepd, \
                 np.arange(1,len(timpd)), \
                 interval=1,blit=False,init_func=init)
        ani_apd.new_frame_seq() 
        
        
    elif event=='Returnpdpso':
        window['respd'].update(visible=False)
        window['pfpd'].update(visible=True)
        
        ax21.cla()
        ax22.cla()
        ax23.cla()
        ax24.cla()
        ax25.cla()
        ax26.cla()
      
        ani_apd.event_source.stop()
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    elif event=='Homesimupdpso':
        window['respd'].update(visible=False)
        window['Home'].update(visible=True)
        ax30.cla()
        ax21.cla()
        ax22.cla()
        ax23.cla()
        ax24.cla()
        ax25.cla()
        ax6.cla()
        ani_apd.event_source.stop()
        ani_apd.new_frame_seq() 
        window['Tablpd'].update(values=tapdv)
        plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)


    #----------------------------------------------------------------------------------------------
      
window.close()
