# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:20:37 2022

@author: marco
"""

import numpy as np
import random
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

# -----------------Péndulo invertido----------------
limit = [(0, 10), (0, 5), (0, 5), (0, 10), (0, 5),
         (0, 5)]       # Limites inferior y superior
pop = 100                    # Tamaño de la población, mayor >= 6
gen = 1000                 # Número de generaciones
D = 6                             # Dimensionalidad O número de variables de diseño
M = 2                              # Numero de objetivos
AMAX = 30
eta = 1
pardyna = [0.5, 0.7, 1, 0.3, 0.05, 0.06, 0.006, np.pi/2, 0]


def inverted_pendulum(r, dimpi):
    '''Time parameters'''
    dt = 0.005  # Tiempo de muestreo (5ms)
    ti = 0.0  # Tiempo inicial de la simulación (0s)
    tf = 15.0  # Tiempo inicial de la simulación (10s)
    n = int((tf - ti) / dt) + 1  # Número de muestras
    # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)
    t = np.linspace(ti, tf, n)

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
        e_x = dimpi[8] - x  # Error de posición de carro
        e_x_dot = 0 - x_dot  # Error de velocidad de carro
        e_th = dimpi[7]-th  # Error de posicón angular
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

    return np.array([ise_next, iadu_next]), g, z, u, t

# -----------------------------------------------------------------------------


# ------------------------------------------------------------------
def dominates(_a, _b):
    for _j in range(M):  # Recorre el vector J de funciones objetivo
        if _b[_j] < _a[_j]:
            return False  # Regresa False si a domina b, en este caso seleccionamos b
    return True  # Regresa Trux si b domina a, en este caso seleccionamos a
# ----------------------------------------------------------------------------------------------------

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


def selec(f, g, po, D, M):
    pop_r = np.empty((0, D))
    f_x_r = np.empty((0, M))
    g_x_r = np.empty(0)

    for r, g_x_i in enumerate(g):
        if g_x_i == 0:
            f_x_r = np.append(f_x_r, [f[r]], axis=0)
            pop_r = np.append(pop_r, [po[r]], axis=0)
            g_x_r = np.append(g_x_r, [g[r]], axis=0)

    f_x_f = np.empty((0, M))  # Conjunto no dominado
    pop_x_f = np.empty((0, D))  # Conjunto no dominado
    g_x_f = np.empty(0)
    # print(len(f_x_r))

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
     #       print(i1)
            g_x_f = np.append(g_x_f, [g_x_r[i1]], axis=0)

    pop_x_r = pop_x_f
    f_x_r = f_x_f
    g_x_r = g_x_f
    return f_x_r, pop_x_r, g_x_r
# ------------------------------------------------------------------------------------


def crossov(p1, p2, eta, llo, lup):
    esp = 1e-14

    for i, (x1, x2) in enumerate(zip(p1, p2)):
        rand = random.random()
        if rand <= 0.5:
            if(abs(p1[i]-p2[i]) > esp):
                if(p1[i] < p2[i]):
                    y1 = p1[i]
                    y2 = p2[i]
                else:
                    y1 = p2[i]
                    y2 = p1[i]
                lb = llo[i]
                up = lup[i]
                ran = random.random()
                beta = 1.0 + ((2. * (y1-lb))/(y2-y1))
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
                    betaq = (1.0 / (2.0 - ran * alpha))**(1.0 / (eta + 1.0))
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
                    p1[i] = c2
                    p2[i] = c1
                else:
                    p1[i] = c1
                    p2[i] = c2
            else:
                p1[i] = p1[i]
                p2[i] = p2[i]
        else:
            p1[i] = p1[i]
            p2[i] = p2[i]

    return p1, p2
# ------------------------------------------------------------------------------------


def mutPolynomial(individual, eta, lb, up, D):
    size = len(individual)
    pm = 1/D

    for i in range(size):
        for k in range(D):
            if random.random() <= pm:
                x = individual[i][k]
                yl = lb[i][k]
                yu = up[i][k]
                if yl == yu:
                    x = yl
                else:
                    delta1 = (x - yl) / (yu - yl)
                    delta2 = (yu - x) / (yu - yl)
                    ra = random.random()
                    mut_pow = 1.0 / (eta + 1.)
                    if ra <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * ra + (1.0 - 2.0 * ra) * ((xy)**(eta + 1.0))
                        delta_q = (val)**(mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - ra) + 2.0 * \
                            (ra - 0.5) * ((xy)**(eta + 1.0))

                        delta_q = 1.0 - (val)**(mut_pow)
                    x = x + delta_q * (yu - yl)
                    individual[i][k] = x
    return np.array(individual)


def moga(limites, poblacion, eta, generaciones, D, M, AMAX, function, pardyna):

    #-----Poblacion------------------------------------------------------------#
    population = np.zeros((gen, pop, D))  # poblacion actual
    population_next = np.zeros((gen, pop, D))  # poblacion siguiente
    # ---------------------------------------------------------------------------
    #------------------F(x)---------------------------------------------------#
    # Valor de funcion objetivo de poblacion actual
    f_x = np.zeros((gen, pop, M))
    # Valor de funcion objetivo de poblacion siguiente
    f_x_next = np.zeros((gen, pop, M))
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Valor de violacion de restricciones de poblacion actual
    g_x = np.zeros((gen, pop))
    # Valor de violacion de restricciones de poblacion siguiente
    g_x_next = np.zeros((gen, pop))
    a = np.empty((0, D))  # Archivo
    # Valor de funcion objetivo para cada elemento del archivo
    f_a = np.empty((0, M))
    # Valor de funcion objetivo para cada elemento del archivo
    g_a = np.empty(0)
    # ---------------------------------------------------------------------------

    li = np.array(limit)
    # Inicializa poblacion
    population[0] = li[:, 0] + np.random.rand(pop, D) * (li[:, 1] - li[:, 0])
    population_next[0] = li[:, 0] + \
        np.random.rand(pop, D) * (li[:, 1] - li[:, 0])

    # -------------Evaluación población 0----------------------------------------------------------------
    for i, xi in enumerate(population[0, :]):  # Evalua objetivos
        solu = function(xi, pardyna)
        f_x[0][i], g_x[0][i] = solu[0], solu[1]  # function(xi,pardyna)
        # ------------------------------------------------------------------------------------------------
    for i in range(0, gen-1):
        f_x_next[i][:] = f_x[i][:]
        population_next[i][:] = population[i][:]
        g_x_next[i][:] = g_x[i][:]

        #print ('Generación:',i)
        selecc = selec(f_x[i, :], g_x[i, :], population[i], D, M)
        f_x_s = selecc[0]
        popu_x_s = selecc[1]
        g_x_s = selecc[2]

        cross = []
        if len(f_x_s) % 2 != 0:
            r1 = random.randint(0, len(popu_x_s)-1)
            p1 = popu_x_s[r1, :]

        lb = np.zeros((len(f_x_s), D))
        up = np.ones((len(f_x_s), D))
        for j in range(math.floor(len(popu_x_s)/2)):

            r1 = j
            r2 = j
            while r1 == j:
                r1 = random.randint(0, len(popu_x_s)-1)

            while r2 == r1 or r2 == j:
                r2 = random.randint(0, len(popu_x_s)-1)
            p1 = popu_x_s[r1, :]
            p2 = popu_x_s[r2, :]

            c = crossov(p1, p2, eta, lb[j], up[j])
            cross.append(c[0])
            cross.append(c[1])
        cro = np.array(cross)
        mut = mutPolynomial(cro, eta, lb, up, D)
        f_x_off = np.zeros((len(mut), M))
        g_x_off = np.zeros(len(mut))

        for r in range(len(mut)):
            mut[r] = asegurar_limites(mut[r], limit)
            val = function(mut[r], pardyna)
            f_x_off[r] = val[0]
            g_x_off[r] = val[1]

            # -------------------------Caso 1-----------------------------------------
            flag_ui = True
            if g_x_off[r] == 0 and g_x[i][r] == 0:
                # Selecciona el individuo que pasa a la siguiente generacion
                if dominates(f_x_off[r], f_x[i][r]):
                    flag_ui = True
                elif dominates(f_x[i][r], f_x_off[r]):
                    flag_ui = False
                else:
                    if random.uniform(0, 1) < 0.5:
                        flag_ui = True
                    else:
                        flag_ui = False
            elif g_x_off[r] > g_x[i][r]:
                flag_ui = False
            elif g_x_off[r] < g_x[i][r]:
                flag_ui = True
            else:
                if random.uniform(0, 1) < 0.5:
                    flag_ui = True
                else:
                    flag_ui = False
            if flag_ui:
                f_x_next[i][r] = np.copy(f_x_off[r])
                population_next[i][r] = np.copy(mut[r])
                g_x_next[i][r] = np.copy(g_x_off[r])
            else:
                f_x_next[i][r] = np.copy(f_x[i][r])
                population_next[i][r] = np.copy(population[i][r])
                g_x_next[i][r] = np.copy(g_x[i][r])

        # Una vez que termina la generacion actualizo x y f_x
        f_x[i+1] = np.copy(f_x_next[i])
        population[i+1] = np.copy(population_next[i])
        g_x[i+1] = np.copy(g_x_next[i])

        # -------------------------Archivo--------------------------------------------------------------------
        # Actualiza archivo (unicamente con soluciones factibles)
        for k, g_x_i in enumerate(g_x[i+1, :]):
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
    plt.figure(1)
    plt.title('Aproximacion al frente de Pareto')

    plt.scatter(f_a[:, 0], f_a[:, 1])
    # plt.xlim([0,1])

    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()
    return f_a, a


Hvmopide = np.zeros(30)

for k in range(30):
    print("Ejecucion "+str(k))
    var = moga(limit, pop, eta, gen, D, M, AMAX, inverted_pendulum, pardyna)
    t = var[0]
    x = t[:, 0]
    y = t[:, 1]

    x = np.sort(x)
    y = np.sort(y)[::-1]
    # print(y)
    x_max = 20
    y_max = 1
    yd = 0
    area2 = 0
    for i in range(len(x)):
        if i == 0:  # primer elemento
            yd = 0

            area2 = 0

            # x_d=x[i+1]-x[i]
            y_d = y_max-y[i]
            # yd=y_d
            # print(yd)
            # area=x_d*yd
            x_d2 = x_max-x[i]
            area2 = x_d2*y_d
            # print(area)
            # print(area2)
        elif (0 < i < len(x)-1):

            # x_d=x[i+1]-x[i]
            y_d = y[i-1]-y[i]
            x_d2 = x_max-x[i]
            # yd=y_d+yd
            area2 = area2+(y_d*x_d2)
            # area=area+(yd*x_d)

        elif i == len(x)-1:  # ultimo elemento

            # x_d1=x[i]-x[i-1]
            y_d = y[i-1]-y[i]
            # area=area+(y_d*x_d1)
            # yd=yd+y_d
            # x_d2=x_max-x[i]
            x_d3 = x_max-x[i-1]
            area2 = area2+(y_d*x_d3)

            print('Hipervolumen:')
            print(area2)
        Hvmopide[k] = area2


filename = "Hvolmogapide.csv"
myFile = open(filename, 'w')
myFile.write("Hv \n")
for l in range(len(Hvmopide)):
    myFile.write(str(Hvmopide[l])+"\n")
myFile.close()
