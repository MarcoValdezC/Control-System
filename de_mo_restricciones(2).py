import numpy as np
import matplotlib.pyplot as plt
from numpy import random

FR = 0.5  # Factor de escalamiento
CR = 0.5  # Factor de cruza
NP = 50  # Tamanio de poblacion
GMAX = 500  # Generacion maxima
AMAX = 100  # Numero maximo de soluciones en el archivo
D = 2  # Dimensionalidad
M = 2  # Numero de objetivos
DMIN = np.array([0, 0])  # Valor minimo de variables de disenio
DMAX = np.array([3, 5])  # Valor maximo de variables de disenio

x = np.zeros((NP, D))  # Poblacion actual
x_next = np.zeros((NP, D))  # Poblacion siguiente

f_x = np.zeros((NP, M))  # Valor de funcion objetivo de poblacion actual
f_x_next = np.zeros((NP, M))  # Valor de funcion objetivo de poblacion siguiente

g_x = np.zeros(NP)  # Valor de violacion de restricciones de poblacion actual
g_x_next = np.zeros(NP)  # Valor de violacion de restricciones de poblacion siguiente

a = np.empty((0, D))  # Archivo
f_a = np.empty((0, M))  # Valor de funcion objetivo para cada elemento del archivo
g_a = np.empty(0)  # Valor de funcion objetivo para cada elemento del archivo


def dominates(_a, _b):
    for _j in range(M):
        if _b[_j] < _a[_j]:
            return False
    return True



def F(_p):
    _x = _p[0]
    _y = _p[1]

    f1 = 4 * _x ** 2 + 4 * _y ** 2
    f2 = (_x - 5) ** 2 + (_y - 5) ** 2

    g = 0
    g += 0 if (_x - 5) ** 2 + _y ** 2 <= 25 else 1
    g += 0 if (_x - 8) ** 2 + (_y + 3) ** 2 >= 7.7 else 1

    return np.array([f1, f2]), g
'''


def F(_p):
    _x = _p[0]
    _y = _p[1]

    f1 = _x
    f2 = (1 + _y) / _x

    g = 0
    g += 0 if _y + 9 * _x >= 6 else 1
    g += 0 if -_y + 9 * _x >= 1 else 1

    return np.array([f1, f2]), g
'''

'''
def F2(_p):
    _x = _p[0]
    _y = _p[1]

    f1 = 4 * _x ** 2 + 4 * _y ** 2
    f2 = (_x - 5) ** 2 + (_y - 5) ** 2

    g1 = (_x - 5) ** 2 + _y ** 2 <= 25
    g2 = (_x - 8) ** 2 + (_y + 3) ** 2 >= 7.7
    g = int(g1) + int(g2)

    return np.array([f1, f2]), g

# x = np.array([-1,0])
x = np.array([1, 1])
f_x, g_x = F2(np.array(x))
print(f'f_x = {f_x}, g_x = {g_x}')
exit(0)
'''

# Paso 1. Inicializacion
x = DMIN + np.random.rand(NP, D) * (DMAX - DMIN)  # Inicializa poblacion

for i, xi in enumerate(x):  # Evalua objetivos
    # print(f'Individuo {i}: {xi}')
    f_x[i], g_x[i] = F(xi)
    # print(g_x[i])
    # print(f_x[i])
    # print('----')

# Paso 2. Ciclo evolutivo
for gen in range(GMAX):  # Para cada generacion
    print('gen = ', gen)
    for i in range(NP):  # Para cada individuo
        # Selecciona r1 != r2 != r3 != i
        r1 = i
        r2 = i
        r3 = i

        while r1 == i:
            r1 = random.randint(0, NP)

        while r2 == r1 or r2 == i:
            r2 = random.randint(0, NP)

        while r3 == r2 or r3 == r1 or r3 == i:
            r3 = random.randint(0, NP)

        # print(f'r1 = {r1}, r2 = {r2}, r3 = {r3}, i = {i}')

        # Genera individuo mutante
        vi = x[r1] + FR * (x[r2] - x[r3])
        # print('vi = ', vi)

        # Genera individuo descendiente
        ui = np.copy(x[i])

        jrand = random.randint(0, D)
        # print(jrand)

        for j in range(D):
            if random.uniform(0, 1) < CR or j == jrand:
                ui[j] = vi[j]

                if ui[j] < DMIN[j]:
                    ui[j] = DMIN[j]

                if ui[j] > DMAX[j]:
                    ui[j] = DMAX[j]
        # Evalua descendiente
        f_ui, g_ui = F(ui)

        # Selecciona el individuo que pasa a la siguiente generacion
        ui_is_better = True
        if g_ui == 0 and g_x[i] == 0:  # Ambas soluciones son factibles
            if dominates(f_ui, f_x[i]):
                ui_is_better = True
            elif dominates(f_x[i], f_ui):
                ui_is_better = False
            else:
                if random.uniform(0, 1) < 0.5:
                    ui_is_better = True
                else:
                    ui_is_better = False
        elif g_ui > g_x[i]: # Hijo viola mas restricciones que padre
            ui_is_better = False
        elif g_ui < g_x[i]: # Padre viola mas restricciones que hijo
            ui_is_better = True
        else: # Ambos violan la misma cantidad de restricciones
            if random.uniform(0, 1) < 0.5:
                ui_is_better = True
            else:
                ui_is_better = False

        if ui_is_better:
            f_x_next[i] = np.copy(f_ui)
            g_x_next[i] = np.copy(g_ui)
            x_next[i] = np.copy(ui)
        else:
            f_x_next[i] = np.copy(f_x[i])
            g_x_next[i] = np.copy(g_x[i])
            x_next[i] = np.copy(x[i])

    # Una vez que termina la generacion actualizo x, f_x, g_x
    f_x = np.copy(f_x_next)
    g_x = np.copy(g_x_next)
    x = np.copy(x_next)

    '''ARCHIVO'''
    # Actualiza archivo (unicamente con soluciones factibles)
    for i, g_x_i in enumerate(g_x):
        if g_x_i == 0:
            f_a = np.append(f_a, [f_x[i]], axis=0)
            a = np.append(a, [x[i]], axis=0)

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
