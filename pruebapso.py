# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 08:07:45 2022

@author: marco
"""

import numpy as np
import matplotlib.pyplot as plt
from random import random
import random
import math

pop=100
gen=50
limit=[(0,10),(0,10),(0,10)] 
D = 3                              
M = 2   
AMAX=30
Vmax=0.1
Vmin=0.0
c1=2
c2=2
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
pardyna=[0.5,1,0.3,0.05,0.006,np.pi]

#----------Problema de optimización---------
def pendulum_s(r,dyna):
    '''Time Parameters'''
    dt = 0.005  # Tiempo de muestreo (5ms)
    ti = 0.0  # Tiempo inicial de la simulación (0s)
    tf = 10.0  # Tiempo inicial de la simulación (10s)
    n = int((tf - ti) / dt) + 1  # Número de muestras
    t = np.linspace(ti, tf, n)  # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)
    
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
        if(ise>=3):
            ie=3
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
def s_best(x,f,g,xbest,fxbest,gxbest):
    x_b=np.zeros((pop,D))
    f_xb=np.zeros((pop,M))
    g_xb=np.zeros(pop)
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
        elif g[u] > gxbest[u]: # Hijo viola mas restricciones que padre
            ui_is_better = False
        elif g[u] < gxbest[u]: # Padre viola mas restricciones que hijo
            ui_is_better = True
        else: # Ambos violan la misma cantidad de restricciones
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
    return x_b,f_xb,g_xb

#---------------------------------------------------------------------------------------------------
def selec(f,g,po):
    pop_r=np.empty((0,D))
    f_x_r=np.empty((0,M))
    g_x_r=np.empty(0)
    
    for r, g_x_i in enumerate(g):
        if g_x_i == 0:
            f_x_r = np.append(f_x_r, [f[r]], axis=0)
            pop_r = np.append(pop_r, [po[r]], axis=0)
            g_x_r=np.append(g_x_r,[g[r]],axis=0)
        
    f_x_f = np.empty((0, M))  # Conjunto no dominado
    pop_x_f = np.empty((0, D))  # Conjunto no dominado
    g_x_f=np.empty(0)
    #print(len(f_x_r))

    for i1, f_a_1 in enumerate(f_x_r):
        sol_nd = True
        for i2, f_a_2 in enumerate(f_x_r):
            if i1 != i2:
                if dominates(f_a_2, f_a_1):
                    sol_nd = False
                    break
        if sol_nd:
            # f_x_fil.append(f_x_1)
            f_x_f = np.append(f_x_f, [f_a_1], axis=0)
            pop_x_f = np.append(pop_x_f, [pop_r[i1]], axis=0)
            g_x_f=np.append(g_x_f,[g_x_r[i1]],axis=0)
    #print(f_x_f)
    best=0
    for i in range(1,len(f_x_f)):
        if dominates(f_x_f[i],f_x_f[best]):
            best=i
    if best==0:
        best=random.randint(0, len(f_x_f)-1)
    #print(best)
   
    pop_x_r= pop_x_f[best]
    f_x_r = f_x_f[best]
    g_x_r = g_x_f[best]
    return f_x_r,pop_x_r,g_x_r

#-----Poblacion------------------------------------------------------------#
population =  np.zeros((gen,pop, D)) #poblacion actual
population_next= np.zeros((gen,pop, D)) #poblacion siguiente 
x_best=np.zeros((gen,pop, D))#Matriz de mejores posiciones locales
x_best_swarp=np.zeros(D)#Gbest
#---------------------------------------------------------------------------
#------------------F(x)---------------------------------------------------#
f_x = np.zeros((gen,pop, M))  # Valor de funcion objetivo de poblacion actual
f_x_next = np.zeros((gen,pop, M))  # Valor de funcion objetivo de poblacion siguiente
f_x_best=np.zeros((gen,pop, M))#Valor de función objetivo
f_x_best_swarp=np.zeros(D)#Gbest
#---------------------------------------------------------------------------
    
#---------------------------------------------------------------------------
g_x = np.zeros((gen,pop))  # Valor de violacion de restricciones de poblacion actual
g_x_next = np.zeros((gen,pop))  # Valor de violacion de restricciones de poblacion siguiente
g_x_best=np.zeros((gen,pop))#Valor de restricciones
g_x_best_swarp=np.zeros(D)#Mejor posición global
a = np.empty((0, D))  # Archivo
f_a = np.empty((0, M))  # Valor de funcion objetivo para cada elemento del archivo
g_a = np.empty(0)  # Valor de funcion objetivo para cada elemento del archivo
#---------------------------------------------------------------------------
vel=np.zeros((pop, D))#VECTOR DE VELOCIDADES


li=np.array(limit)
population[0]=li[:,0] + np.random.rand(pop, D) * (li[:,1] - li[:,0])  # Inicializa poblacion
x_best[0]=population[0]



#-------------Evaluación población 0------------------------------------------------------------------
for i, xi in enumerate(population[0,:]):  # Evalua objetivos
    solu=pendulum_s(xi,pardyna)
    f_x[0][i], g_x[0][i] =solu[0],solu[1] #function(xi,pardyna)
    #------------------------------------------------------------------------------------------------------
f_x_best[0]=f_x[0]
g_x_best[0]=g_x[0]

selecc=selec(f_x[0,:],g_x[0,:],population[0])
f_x_best_swarp=selecc[0]
x_best_swarp=selecc[1]
g_x_best_swarp=selecc[2]

for i in range(0,gen-1):
    
    print ('Generación:',i) 
    w=Vmax-(i /gen)*(Vmax-Vmin)
    r1=random.random()
    r2=random.random()
    for j in range(pop):
        vel[j]=w*vel[j]+ r1*c1*(x_best[i][j]-population[i][j])+r2*c2*(x_best_swarp-population[i][j])
    population_next[i]=population[i]+vel
    for h in range(pop):
        population_next[i][h]=asegurar_limites(population_next[i][h],limit)
        sol=pendulum_s(population_next[i][h], pardyna)
        f_x_next[i][h],g_x_next[i][h]=sol[0],sol[1]
    population[i+1]=population_next[i]
    f_x[i+1]=f_x_next[i]
    g_x[i+1]=g_x_next[i]
    sele=s_best(population[i+1],f_x[i+1],g_x[i+1],x_best[i],f_x_best[i],g_x_best[i])
    x_best[i+1]=sele[0]
    f_x_best[i+1]=sele[1]
    g_x_best[i+1]=sele[2]
    selecc=selec(f_x_best[i+1],g_x_best[i+1],x_best[i+1])
    f_x_best_swarp=selecc[0]
    x_best_swarp=selecc[1]
    g_x_best_swarp=selecc[2]

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

plt.figure(1)
plt.title('Aproximacion al frente de Pareto')

plt.scatter(f_a[:, 0], f_a[:, 1])
#plt.xlim([0,1])

plt.xlabel('f1')
plt.ylabel('f2')
plt.show()
    
    
    
    
        
    
   
    
    
    
    