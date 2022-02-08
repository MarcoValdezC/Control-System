    # -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:17:44 2022

@author: marco
"""

import PySimpleGUI as sg
import math 
import numpy as np 
import random
import numpy as np 
import math
import matplotlib.pyplot as plt
from drawnow import *
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from numpy.linalg import inv


#---------------------Parametros DEPS-----------------------------#
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
    # print(a)
    # print(f_a)
    # #-------Guardar en archivo excel-----------------------------------------
  
    # filename="afa.csv" 
    # myFile=open(filename,'w') 
    # myFile.write("kp,kd,ki,f1, f2 \n") 
    # for l in range(len(f_a)): 
    #     myFile.write(str(a[l, 0])+","+str(a[l, 1])+","+str(a[l, 2])+","+str(f_a[l, 0])+","+str(f_a[l, 1])+"\n") 
    # myFile.close()
    # #------------Gráfica del Frente de Pareto-----------------------
    # plt.figure(1)
    # plt.title('Aproximacion al frente de Pareto')
    # plt.scatter(f_a[:, 0], f_a[:, 1])
    # plt.xlabel('f1')
    # plt.ylabel('f2')
    # plt.show()
    
    return f_a,a


#-----------------Péndulo invertido----------------
limitpi=[(0,10),(0,10),(0,10),(0,10),(0,10),(0,10)]       # Limites inferior y superior
poblacionpi = 200                    # Tamaño de la población, mayor >= 4
f_mutpi = 0.5                        # Factor de mutacion [0,2]
recombinationpi = 0.7                # Tasa de  recombinacion [0,1]
generacionespi =10                 # Número de generaciones
Dpi = 6                             # Dimensionalidad O número de variables de diseño 
Mpi = 2                              # Numero de objetivos
AMAXpi = 30     

def inverted_pendulum(r,dimpi):
    
    '''Time parameters'''
    dt = 0.005  # Tiempo de muestreo (5ms)
    ti = 0.0  # Tiempo inicial de la simulación (0s)
    tf = 10.0  # Tiempo inicial de la simulación (10s)
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
        if(iadu>=2):
            ia=2
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


#Function for drawing
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


##-----DEFAULT SETTINGS----------------------------------##
bw: dict = {'size': (20, 20), 'font': ('Franklin Gothic Book', 60), 'button_color': ("blue", "#F8F8F8")}
bt: dict = {'size': (12, 1), 'font': ('Franklin Gothic Book', 18), 'button_color': ("black", "#F1EABC")}
bo: dict = {'size': (15, 2), 'font': ('Arial', 24), 'button_color': ("black", "#ECA527"), 'focus': True}

layouthome= [[sg.Text('CONTROL PID CON OPTIMIZACIÓN MULTIOBJETIVO',justification='center', 
             text_color='white', font=('Franklin Gothic Book', 28, 'bold'))],
             [sg.Text('Selecciona un péndulo:', justification='center',text_color='white', font=('Franklin Gothic Book', 14, 'bold'))],
             [sg.Button(image_filename='D:\TT2\imagen.png' ,key='Simple',button_color=(sg.theme_background_color(), sg.theme_background_color())), sg.Button(image_filename='D:\TT2\dob.png', key='Invertido',button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button(image_filename='D:\TT2\doble.png', key='Doble')],
             [sg.Text('Simple',size=(38,2),justification='center',font=('Franklin Gothic Book', 15, 'bold')), sg.Text('Invertido',size=(38,2), justification='center',font=('Franklin Gothic Book', 15, 'bold')),sg.Text('Doble',size=(38,2),justification='center',font=('Franklin Gothic Book', 15, 'bold'))],
             [sg.Button('Salir',button_color='red',size=(5,2),border_width=5,key='Exit')]]

layouts=[[sg.Text('Péndulo Simple:',text_color='white', font=('Franklin Gothic Book', 28, 'bold')) ],
            [sg.Text('Masa (m):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.Input('',key='masaps')],
            [sg.Text('Longitud (l):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('',key='lps')],
            [sg.Text('Longitud al centro masa (lc):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('',key='lcps')],
            [sg.Text('Fricción (D):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText(key='bps')],
            [sg.Text('Momento de inercia (I):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('',key='ips')],
            [sg.Text('Set point (rad):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('',key='sps')],
            [sg.Text('Seleccione el algoritmo metaheurístico:',text_color='white', font=('Franklin Gothic Book', 28, 'bold'))],
            [sg.Button('DE',button_color='blue',border_width=3,key='deps'),sg.Button('GE',button_color='blue',border_width=3,key='geps'),sg.Button('PSO',button_color='blue',border_width=3,key='psops')],
            [sg.Button(image_filename='D:\TT2\home.png', key='Homeps',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color()))]
            ]

layouti=[[sg.Text('Péndulo Invertido:',text_color='white', font=('Franklin Gothic Book', 28, 'bold')) ],
            [sg.Text('Masa del péndulo(m):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.Input('',key='masapi')],
            [sg.Text('Masa del carrito(M):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.Input('',key='masaca')],
            [sg.Text('Longitud (l):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('',key='lpi')],
            [sg.Text('Longitud al centro masa (lc):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('',key='lcpi')],
            [sg.Text('Fricción del péndulo(b1):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText(key='bpi')],
            [sg.Text('Fricción del carrito(b2):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText(key='bca')],
            [sg.Text('Momento de inercia (I):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('',key='ipi')],
            [sg.Text('Set point del péndulo (rad):', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('',key='spi')],
            [sg.Text('Set point del carrito:', text_color='black', font=('Franklin Gothic Book', 12, 'bold'),size=(24,1)), sg.InputText('',key='spc')],
            [sg.Text('Seleccione el algoritmo metaheurístico:',text_color='white', font=('Franklin Gothic Book', 28, 'bold'))],
            [sg.Button('DE',button_color='blue',border_width=3,key='depi'),sg.Button('GE',button_color='blue',border_width=3,key='gepi'),sg.Button('PSO',button_color='blue',border_width=3,key='psopi')],
            [sg.Button(image_filename='D:\TT2\home.png', key='Homepi',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color()))]
            ]

layoutpfps=[[sg.Text('Seleccione un conjunto de ganancias del controlador  PID',text_color='white', font=('Franklin Gothic Book', 20, 'bold'))],
            [sg.Table(values=[['Espera','estamos','ejecutando','código','......'],['Espera','estamos','ejecutando','código','......']] ,headings=['Kp' , 'Kd' ,' Ki','ISE','IADU'],auto_size_columns=True,right_click_selects=True,enable_click_events=True, key='Tabl',vertical_scroll_only=False,num_rows=25 ), sg.Canvas(key='can')],
            [sg.Button('Simular',key='Simups')]]

layoutpfpi=[[sg.Text('Seleccione un conjunto de ganancias del controlador  PID',text_color='white', font=('Franklin Gothic Book', 20, 'bold'))],
            [sg.Table(values=[['Espera','estamos','ejecutando','código','......','......','......','......'],['Espera','estamos','ejecutando','código','......','......','......','......']] ,headings=['Kpcar' , 'Kdcar' ,'Kicar','Kppénd' , 'Kdpén' ,' Kipén','ISE','IADU'],auto_size_columns=True,right_click_selects=True,enable_click_events=True, key='Tablpi',vertical_scroll_only=False,num_rows=25 ), sg.Canvas(key='canpfpi')],
            [sg.Button('Simular',key='Simupi')]]


layoutsimpan=[[sg.Canvas(key='canani')]]
layoutsimpgra=[[sg.Canvas(key='cangraps')]]

layouttap=[[sg.TabGroup([[sg.Tab('Animación',layoutsimpan),sg.Tab('Graficas', layoutsimpgra)]],tab_location='centertop',border_width=5)],
           [sg.Button(image_filename='D:\TT2\home.png', key='Homesimups',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Salir',button_color='red',size=(3,2),border_width=5,key='Exit0')]]

layoutinvpan=[[sg.Canvas(key='cananipi')]]
layoutinvpgra=[[sg.Canvas(key='cangrapi')]]

layouttappi=[[sg.TabGroup([[sg.Tab('Animación',layoutinvpan),sg.Tab('Graficas', layoutinvpgra)]],tab_location='centertop',border_width=5)],
           [sg.Button(image_filename='D:\TT2\home.png', key='Homesimupi',image_subsample=8,button_color=(sg.theme_background_color(), sg.theme_background_color())),sg.Button('Salir',button_color='red',size=(3,2),border_width=5,key='Exit1')]]

layout1=[[sg.Column(layouthome,key='Home'),sg.Column(layouts, visible=False,key='Sim'),sg.Column(layoutpfps,key='pfps',visible=False),sg.Column(layouttap,key='resps',visible=False),sg.Column(layouti,key='Inve',visible=False),sg.Column(layoutpfpi,key='pfpi',visible=False),sg.Column(layouttappi,key='respi',visible=False)]]
        
# #Create a fig for embedding.
# fig = plt.figure(figsize=(5, 4))
# ax = fig.add_subplot(111)



window = sg.Window('Swapping the contents of a window', layout1, finalize=True)
#Associate fig with Canvas.

layout = 1  # The currently visible layout
while True:
    event, values = window.read()
    print(event, values)
  

    
    #u=float(p)
    #e=eval(values[0])
    if event in (None, 'Exit','Exit0'):
        break
    if event == 'Simple':
        #print(window['Simple'].get_size())
        window['Home'].update(visible=False)
        window['Sim'].update(visible=True)
    if event == 'deps':
        window['Sim'].update(visible=False)
        window['pfps'].update(visible=True)
        
        ms=values['masaps']
        ls=values['lps']
        lcs=values['lcps']
        bs=values['bps']
        iss=values['ips']
        ss=values['sps']
        
        dinps=np.asarray([ms,ls,lcs,bs,iss,ss], dtype=np.float64, order='C')
        sg.popup('Ejecución de evolución diferencial, espera para poder observar el resultado (Conjunto de ganancias para el controlador PID).Las ganancias permitiran al péndulo llegar de la posición inial a la deseada .Presiona ok para continuar con la ejecución')
        # dina=val_conver(dinps)
       
        #llamado de la función main de DE
        var=main(pendulum_s, limit, poblacion, f_mut, recombination, generaciones,dinps,D,M,AMAX)
        
        valu=np.zeros((len(var[0]),5))
        
        t=var[0]
        s=var[1]
        
        valu[:,0]=s[:,0]
        valu[:,1]=s[:,1]
        valu[:,2]=s[:,2]
        valu[:,3]=t[:,0]
        valu[:,4]=t[:,1]
        
        #Create a fig for embedding.
        fig = plt.figure(figsize=(6, 5))
        
        ax = fig.add_subplot(111)
        ax.set_title('Aproximación al frente de Pareto')
        ax.set_xlabel('f1')
        ax.set_ylabel('f2')
        
        #plot
        ax.scatter(t[:,0], t[:,1])
        
        fig_agg = draw_figure(window['can'].TKCanvas, fig)
        
        window['Tabl'].update(values=valu)
        #After making changes, fig_agg.draw()Reflect the change with.
        fig_agg.draw()
       
    elif event=='Homeps':
        window['Sim'].update(visible=False)
        window['Home'].update(visible=True)
    elif event=='Simups':
        window['pfps'].update(visible=False)
        window['resps'].update(visible=True)
        
        afe=values['Tabl']
        #print(s[afe[0],:])
        pen=pendulum_s(s[afe[0],:], dinps)
        posi=pen[2]
        tor=pen[3]
        tim=pen[4]
        
        figan = plt.figure(figsize=(7, 6))
        ax1 = figan.add_subplot(111, autoscale_on=False,xlim=(-1.8, 1.8), ylim=(-1.2, 1.2))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        fig_anima = draw_figure(window['canani'].TKCanvas, figan)
        
      
        
        figgraps = plt.figure(figsize=(7, 6))
        ax2 = figgraps.add_subplot(221)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Posición del péndulo')
        ax2.plot(tim, posi[:, 0], 'k',label=r'$\theta$',lw=1)
        ax2.legend()
        
        ax3 = figgraps.add_subplot(222)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Velocidad del péndulo')
        ax3.plot(tim, posi[:, 1], 'b',label=r'$\dot{\theta}$',lw=1)
        ax3.legend()
        
        ax4 = figgraps.add_subplot(223)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Señal de control')
        ax4.plot(tim, tor[:, 0], 'r',label=r'$u$',lw=1)
        ax4.legend()
        
        
        fig_graps = draw_figure(window['cangraps'].TKCanvas, figgraps)
        
        
        x0 = np.zeros(len(tim))
        y0 = np.zeros(len(tim))
        
        l=dinps[1]
     
        x1 = l * np.sin(posi[:, 0])
        y1 = -l * np.cos(posi[:, 0])
        line, = ax1.plot([], [], 'o-', color='orange', lw=4, \
                markersize=6, markeredgecolor='k', \
                markerfacecolor='k')
        time_template = 't= %.1fs'
        time_text = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)
        def init():
            
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text
        def animate(i):
            line.set_data([x0[i], x1[i]], [y0[i], y1[i]])
            time_text.set_text(time_template % tim[i])
            return line, time_text,
        ani_a = animation.FuncAnimation(figan, animate, \
                                np.arange(1, len(tim)), \
                                interval=40, blit=False)
            
    elif event=='Homesimups':
        window['resps'].update(visible=False)
        window['Home'].update(visible=True)
    elif event=='Invertido':
        window['Home'].update(visible=False)
        window['Inve'].update(visible=True)
    elif event == 'depi':
        window['Inve'].update(visible=False)
        window['pfpi'].update(visible=True)
        
        mi=values['masapi']
        mc=values['masaca']
        li=values['lpi']
        lci=values['lcpi']
        bi=values['bpi']
        bc=values['bca']
        isi=values['ipi']
        sti=values['spi']
        stc=values['spc']
        
        dinpi=np.asarray([mi,mc,li,lci,bi,bc,isi,sti,stc], dtype=np.float64, order='C')
        sg.popup('Ejecución de evolución diferencial, espera para poder observar el resultado (Conjunto de ganancias para el controlador PID).Las ganancias permitiran al péndulo llegar de la posición inial a la deseada .Presiona ok para continuar con la ejecución')
        # dina=val_conver(dinps)
       
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
        
        #Create a fig for embedding.
        figpi = plt.figure(figsize=(6, 5))
        
        ax6 = figpi.add_subplot(111)
        ax6.set_title('Aproximación al frente de Pareto')
        ax6.set_xlabel('f1')
        ax6.set_ylabel('f2')
        
        #plot
        ax6.scatter(tpi[:,0], tpi[:,1])
        
        fig_aggpi = draw_figure(window['canpfpi'].TKCanvas, figpi)
        
        window['Tablpi'].update(values=valupi)
        #After making changes, fig_agg.draw()Reflect the change with.
        fig_aggpi.draw()
    elif event=='Homepi':
        window['Inve'].update(visible=False)
        window['Home'].update(visible=True)

    elif event=='Simupi':
        window['pfpi'].update(visible=False)
        window['respi'].update(visible=True)
        
        afepi=values['Tablpi']
        #print(s[afe[0],:])
        penpi=inverted_pendulum(spi[afepi[0],:], dinpi)
        posipi=penpi[2]
        torpi=penpi[3]
        timpi=penpi[4]
        
        figanpi = plt.figure(figsize=(7, 6))
        ax10 = figanpi.add_subplot(111, autoscale_on=False,xlim=(-1.8, 1.8), ylim=(-1.2, 1.2))
        ax10.set_xlabel('x')
        ax10.set_ylabel('y')
        fig_animapi = draw_figure(window['cananipi'].TKCanvas, figanpi)
        
      
        
        figgrapspi = plt.figure(figsize=(7, 6))
        ax11 = figgrapspi.add_subplot(321)
        ax11.set_xlabel('Tiempo')
        ax11.set_ylabel('Posición del carro')
        ax11.plot(timpi, posipi[:, 0], 'k',label=r'$x$',lw=1)
        ax11.legend()
        
        ax12 = figgrapspi.add_subplot(322)
        ax12.set_xlabel('Tiempo')
        ax12.set_ylabel('Posición del péndulo')
        ax12.plot(timpi, posipi[:, 1], 'b',label=r'$\theta$',lw=1)
        ax12.legend()
        
        ax13 = figgrapspi.add_subplot(323)
        ax13.set_xlabel('Tiempo')
        ax13.set_ylabel('Velocidad del carrito')
        ax13.plot(timpi, posipi[:, 2], 'r',label=r'$\dot{x}$',lw=1)
        ax13.legend()
        
        ax14 = figgrapspi.add_subplot(324)
        ax14.set_xlabel('Tiempo')
        ax14.set_ylabel('Velocidad del péndulo')
        ax14.plot(timpi, posipi[:, 3], 'k',label=r'$x$',lw=1)
        ax14.legend()
        
        ax15 = figgrapspi.add_subplot(325)
        ax15.set_xlabel('Tiempo')
        ax15.set_ylabel('$u_{car}$')
        ax15.plot(timpi, torpi[0, :], 'b',label=r'$u_{car}$',lw=1)
        ax15.legend()
        
        ax16 = figgrapspi.add_subplot(326)
        ax16.set_xlabel('Tiempo')
        ax16.set_ylabel('$u_{pendulum}$')
        ax16.plot(timpi, torpi[1, :], 'r',label=r'$u_{pendulum}$',lw=1)
        ax16.legend()
        
        
        fig_graps = draw_figure(window['cangrapi'].TKCanvas, figgrapspi)
        
        
        x1 = posipi[:, 0]
        y1 = np.zeros(len(timpi))
        
        l=dinpi[2]
     
        x2 = l * np.cos(posipi[:, 1]) + x1
        y2 = l * np.sin(posipi[:, 1])
        
        mass1, = ax10.plot([], [], linestyle='None', marker='s', \
                 markersize=10, markeredgecolor='k', \
                 color='green', markeredgewidth=2)
        line, = ax10.plot([], [], 'o-', color='green', lw=4, \
                markersize=6, markeredgecolor='k', \
                markerfacecolor='k')
        time_template = 't= %.1fs'
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
        
        
        ani_api = animation.FuncAnimation(figanpi, animatepi, \
                                np.arange(1, len(timpi)), \
                                interval=40, blit=False)
    
window.close()