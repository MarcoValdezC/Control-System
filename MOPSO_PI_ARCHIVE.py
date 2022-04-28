# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:24:34 2022

@author: marco
"""
import numpy as np
import matplotlib.pyplot as plt
from random import random
import random
import math
from numpy.linalg import inv

#---------------------Parametros PSO-----------------------------#
pop = 100
gen = 1000
limit=[(0,10),(0,10),(0,10),(0,20),(0,20),(0,20)]
D = 6
M = 2
AMAX = 30
Vmax = 0.1
Vmin = 0.0
c1 = 1
c2 = 1
#---------------------End Parametros PSO--------------------------#

#----------------Funcion de dominancia-------#


def dominates(_a, _b):
    for _j in range(M):  # Recorre el vector J de funciones objetivo
        if _b[_j] < _a[_j]:
            return False  # Regresa False si a domina b, en este caso seleccionamos b
    return True  # Regresa Trux si b domina a, en este caso seleccionamos a
#----------------End Funcion de dominancia-------#


pardyna = [0.5, 1, 0.3, 0.05, 0.006, np.pi, D, M]





def inverted_pendulum(r):
    '''Time parameters'''
    dt = 0.005  # Tiempo de muestreo (5ms)
    ti = 0.0  # Tiempo inicial de la simulación (0s)
    tf = 10.0  # Tiempo inicial de la simulación (10s)
    n = int((tf - ti) / dt) + 1  # Número de muestras
    # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)
    t = np.linspace(ti, tf, n)

    '''Dynamic parameters'''
    m = 0.5  # Masa del pendulo (kg)
    M = 0.7  # Masa del carro (kg)
    l = 1.0  # Longitud de la barra del péndulo (m)
    lc = 0.3  # Longitud al centro de masa del péndulo (m)
    b1 = 0.05  # Coeficiente de fricción viscosa pendulo
    b2 = 0.06  # Coeficiente de friccion del carro
    gra = 9.81  # Aceleración de la gravedad en la Tierra
    I = 0.006  # Tensor de inercia del péndulo

    '''State variables'''
    z = np.zeros((n, 4))

    '''Control vector'''
    u = np.zeros((2, n))

    '''Initial conditions'''
    z[0, 0] = 0  # Posición inicial del carro (m)
    z[0, 1] = 0  # Posición inicial del péndulo (rad)
    z[0, 2] = 0  # Velocidad inicial del carro (m/s)
    z[0, 3] = 0  # Velocidad angular del péndulo (rad/s)
    ie_th = 0  # Inicialización de error integral de posición de carro
    ie_x = 0  # Inicialización de error integral de posición de péndulo

    ise = 0
    ise_next = 0
    iadu = 0
    iadu_next = 0

    '''State equation'''
    zdot = [0, 0, 0, 0]

    '''Dynamic simulation'''
    for c in range(n - 1):
        '''Current states'''
        x = z[c, 0]  # Posicion del carro
        th = z[c, 1]   # Posición del péndulo
        x_dot = z[c, 2]  # Velocidad del carro
        th_dot = z[c, 3]  # Velocidad del péndulo

        '''Controller'''
        e_x = 0 - x  # Error de posición de carro
        e_x_dot = 0 - x_dot  # Error de velocidad de carro
        e_th = np.pi/2-th  # Error de posicón angular
        e_th_dot = 0 - th_dot  # Error de velocidad angular

        '''Ganancias del controlador del carro'''
        Kp = r[0]
        Kd = r[1]
        Ki = r[2]

        '''Ganancias del controlador del péndulo'''
        Kp1 = r[3]
        Kd1 = r[4]
        Ki1 = r[5]
        # print(r)

        # Señal de control del actuador del carro
        u[0, c] = Kp * e_x + Kd * e_x_dot + Ki * ie_x
        # Señal de control del actuador del péndulo
        u[1, c] = Kp1 * e_th + Kd1 * e_th_dot + Ki1 * ie_th

        # print(u[0,c])
        # print(u[1,c])

        MI = np.array([[M + m, -m * lc * np.sin(th)], [-m * lc *
                      np.sin(th), I + m * lc ** 2]])  # Matriz de inercia
        # Matriz de Coriollis
        MC = np.array([[b1, -m * lc * np.cos(th) * th_dot], [0, b2]])
        MG = np.array([[0], [m * gra * l * np.cos(th)]])  # Vector de gravedad

        array_dots = np.array([[x_dot], [th_dot]])  # Vector de velocidades
        MC2 = np.dot(MC, array_dots)

        ua = np.array([[u[0, c]], [u[1, c]]])
        aux1 = ua - MC2 - MG
        Minv = inv(MI)
        # Varables de segundo grado /(doble derivada)
        aux2 = np.dot(Minv, aux1)

        '''System dynamics'''
        zdot[0] = x_dot  # Velocidad del carro
        zdot[1] = th_dot  # Velocidad del péndulo
        zdot[2] = aux2[0, :]  # Aceleración del carro
        zdot[3] = aux2[1, :]  # Aceleración del péndulo

        '''Integrate dynamics'''
        z[c + 1, 0] = z[c, 0] + zdot[0] * dt
        z[c + 1, 1] = z[c, 1] + zdot[1] * dt
        z[c + 1, 2] = z[c, 2] + zdot[2] * dt
        z[c + 1, 3] = z[c, 3] + zdot[3] * dt
        ie_th = ie_th + e_th * dt
        ie_x = ie_x+e_x*dt

        ise = ise_next+(e_th**2)*dt+(e_x**2)*dt
        iadu = iadu_next + (abs(u[0, c]-u[0, c-1])) * \
            dt+(abs(u[1, c]-u[1, c-1]))*dt
        g = 0
        if(ise >= 20):
            ie = 20
            g += 1
        else:
            ie = ise
            g += 0
        if(iadu >= 1.2):
            ia = 1.2
            g += 1
        else:
            ia = iadu
            g += 0

        ise_next = ie
        iadu_next = ia
    u[:, n - 1] = u[:, n - 2]  # Actualizar señal de control

    #print(z[:, 0])

    return np.array([ise_next, iadu_next]), g





# ---------------Asegurar limites de caja-------------------------------------------------------------


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


def MOPSO(function, limit, pop, Vmax, Vmin, c1, c2, gen, D, M, AMAX):
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
        solu = function(xi)
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
            sol = function(population_next[i][h])
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



Hvmopsopi=np.zeros(30)

for k in range(30):
    print(k)
    var = MOPSO(inverted_pendulum, limit, pop, Vmax, Vmin,
            c1, c2, gen, D, M, AMAX)
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
           
            #x_d=x[i+1]-x[i]
            y_d=y_max-y[i]
            #yd=y_d
            #print(yd)
            #area=x_d*yd
            x_d2=x_max-x[i]
            area2=x_d2*y_d
            # print(area)
            # print(area2)
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
        Hvmopsopi[k]=area2
    
    
filename="Hvolmopsopi.csv" 
myFile=open(filename,'w') 
myFile.write("Hv \n") 
for l in range(len(Hvmopsopi)): 
    myFile.write(str(Hvmopsopi[l])+"\n")  
myFile.close()