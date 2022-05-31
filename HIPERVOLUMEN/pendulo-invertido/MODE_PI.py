# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:58:33 2022

@author: marco
"""
import os
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import random
import math
from drawnow import *

#---------------------Parametros DE-----------------------------#
limit = [(0, 10), (0, 10), (0, 10), (0, 20), (0, 20),
         (0, 20)]       # Limites inferior y superior
poblacion = 100                    # Tamaño de la población, mayor >= 4
f_mut = 0.5                        # Factor de mutacion [0,2]
recombination = 0.7                # Tasa de  recombinacion [0,1]
generaciones = 1000                 # Número de generaciones
D = 6                             # Dimensionalidad O número de variables de diseño
M = 2                              # Numero de objetivos
AMAX = 30                          # Numero maximo de soluciones en el archivo
# ----------------------------------------------------------------

#---------------Función de dominancia------#


def dominates(_a, _b):
    for _j in range(M):  # Recorre el vector J de funciones objetivo
        if _b[_j] < _a[_j]:
            return False  # Regresa False si a domina b, en este caso seleccionamos b
    return True  # Regresa Trux si b domina a, en este caso seleccionamos a
# ----------------------------------------------------------------------------------------------------


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
        e_th = (np.pi/2)-th  # Error de posicón angular
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
# ---------------------------------------------------------------------------------------------------


def main(function, limites, poblacion, f_mut, recombination, generaciones):

    #-----Poblacion------------------------------------------------------------#
    population = np.zeros((generaciones, poblacion, D))  # poblacion actual
    population_next = np.zeros(
        (generaciones, poblacion, D))  # poblacion siguiente
    # ---------------------------------------------------------------------------

    #------------------F(x)---------------------------------------------------#
    # Valor de funcion objetivo de poblacion actual
    f_x = np.zeros((generaciones, poblacion, M))
    # Valor de funcion objetivo de poblacion siguiente
    f_x_next = np.zeros((generaciones, poblacion, M))
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Valor de violacion de restricciones de poblacion actual
    g_x = np.zeros((generaciones, poblacion))
    # Valor de violacion de restricciones de poblacion siguiente
    g_x_next = np.zeros((generaciones, poblacion))
    a = np.empty((0, D))  # Archivo
    # Valor de funcion objetivo para cada elemento del archivo
    f_a = np.empty((0, M))
    # Valor de funcion objetivo para cada elemento del archivo
    g_a = np.empty(0)
    # ---------------------------------------------------------------------------

    # --------------------Inicialización de la población-------------------------
    for i in range(0, poblacion):  # cambiar tam_poblacion
        indv = []
        for j in range(len(limites)):
            indv.append(random.uniform(limites[j][0], limites[j][1]))
            # print(indv[0])
        population[0][i] = indv[0]
        population_next[0][i] = indv[0]
    # -------------------------------------------------------------------------------

    # -------------Evaluación población 0------------------------------------------------------------------
    for i, xi in enumerate(population[0, :]):  # Evalua objetivos

        f_x[0][i], g_x[0][i] = function(xi)
    # ------------------------------------------------------------------------------------------------------

    # ---------------------Ciclo evolutivo------------------------------------------------------------------
    for i in range(0, generaciones-1):
        #print('Generación:', i)
        for j in range(0, poblacion):

            # Mutacion
            # Seleccionamos 4 posiciones de vector aleatorios, range = [0, poblacion)
            candidatos = range(0, poblacion)
            random_index = random.sample(candidatos, 4)

            r1 = random_index[0]
            r2 = random_index[1]
            r3 = random_index[2]

            while r1 == j:
                t = random.sample(candidatos, 1)
                r1 = t[0]

            while r2 == r1 or r2 == j:
                t2 = random.sample(candidatos, 1)
                r2 = t2[0]

            while r3 == r2 or r3 == r1 or r3 == j:
                t3 = random.sample(candidatos, 1)
                r3 = t3[0]

            x_1 = population[i][r1]
            x_2 = population[i][r2]
            x_3 = population[i][r3]
            x_t = population[i][j]

            # Restamos x3 de x2, y creamos un nuevo vector (x_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

            # Multiplicamos x_diff por el factor de mutacion(F) y sumamos x_1
            v_mutante = [x_1_i + f_mut * x_diff_i for x_1_i,
                         x_diff_i in zip(x_1, x_diff)]
            v_mutante = asegurar_limites(v_mutante, limites)
            # print(v_mutante)
            # Vector hijo
            v_hijo = np.copy(population[i][j])
            jrand = random.randint(0, D)

            for k in range(len(x_t)):
                crossover = random.uniform(0, 1)
                if crossover <= recombination or k == jrand:
                    v_hijo[k] = v_mutante[k]
                else:
                    v_hijo[k] = x_t[k]

            # Evalua descendiente
            f_ui, g_ui = function(v_hijo)

            # -------------------------Caso 1-----------------------------------------
            flag_ui = True
            if g_ui == 0 and g_x[i][j] == 0:
                # Selecciona el individuo que pasa a la siguiente generacion
                if dominates(f_ui, f_x[i][j]):
                    flag_ui = True
                elif dominates(f_x[i][j], f_ui):
                    flag_ui = False
                else:
                    if random.uniform(0, 1) < 0.5:
                        flag_ui = True
                    else:
                        flag_ui = False
            elif g_ui > g_x[i][j]:
                flag_ui = False
            elif g_ui < g_x[i][j]:
                flag_ui = True
            else:
                if random.uniform(0, 1) < 0.5:
                    flag_ui = True
                else:
                    flag_ui = False
            if flag_ui:
                f_x_next[i][j] = np.copy(f_ui)
                population_next[i][j] = np.copy(v_hijo)
                g_x_next[i][j] = np.copy(g_ui)
            else:
                f_x_next[i][j] = np.copy(f_x[i][j])
                population_next[i][j] = np.copy(population[i][j])
                g_x_next[i][j] = np.copy(g_x[i][j])

        # Una vez que termina la generacion actualizo x y f_x
        f_x[i+1] = np.copy(f_x_next[i])
        population[i+1] = np.copy(population_next[i])
        g_x[i+1] = np.copy(g_x_next[i])

    # -------------------------Archivo--------------------------------------------------------------------
        # Actualiza archivo (unicamente con soluciones factibles)
        for r, g_x_i in enumerate(g_x[i+1, :]):
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
        # print(a)
        # print(f_a)
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
    # -------Guardar en archivo excel-----------------------------------------

    # filename="pifa.csv"
    # myFile=open(filename,'w')
    # myFile.write("kp,kd,ki,kp1,kd1,ki1,f1, f2 \n")
    # for l in range(len(f_a)):
    #     myFile.write(str(a[l, 0])+","+str(a[l, 1])+","+str(a[l, 2])+","+str(a[l, 3])+","+str(a[l, 4])+","+str(a[l, 5])+","+str(f_a[l, 0])+","+str(f_a[l, 1])+"\n")
    # myFile.close()
    # #------------Gráfica del Frente de Pareto-----------------------

    return f_a


Hvpide = np.zeros(30)

for r in range(30):
    print("Ejecucion "+str(r))
    # llamado de la función main de DE
    var = main(inverted_pendulum, limit, poblacion,
               f_mut, recombination, generaciones)

    x = var[:, 0]
    y = var[:, 1]
    x = np.sort(x)
    y = np.sort(y)[::-1]
    x_max = 20
    y_max = 1
    yd = 0

    area2 = 0

    for i in range(len(x)):
        if i == 0:  # primer elemento
            yd = 0

            area2 = 0

            y_d = y_max-y[i]

            x_d2 = x_max-x[i]
            area2 = x_d2*y_d

        elif (0 < i < len(x)-1):

            y_d = y[i-1]-y[i]
            x_d2 = x_max-x[i]

            area2 = area2+(y_d*x_d2)

        elif i == len(x)-1:  # ultimo elemento

            y_d = y[i-1]-y[i]
            x_d3 = x_max-x[i-1]
            area2 = area2+(y_d*x_d3)

            print('Hipervolumen:')
            print(area2)
        Hvpide[r] = area2


filename = "Hvolpide.csv"
myFile = open(filename, 'w')
myFile.write("Hv \n")
for l in range(len(Hvpide)):
    myFile.write(str(Hvpide[l])+"\n")
myFile.close()
