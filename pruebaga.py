# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:59:52 2022

@author: marco
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt

pop=100
gen=10
limit=[(0,10),(0,10),(0,10)] 
D = 3                              
M = 2   
AMAX=30
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
#---------------------------------------------------------------------------------------------------

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
     #       print(i1)
            g_x_f=np.append(g_x_f,[g_x_r[i1]],axis=0)
            

    pop_x_r= pop_x_f
    f_x_r = f_x_f
    g_x_r = g_x_f
    return f_x_r,pop_x_r,g_x_r

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
def mutPolynomial(individual, eta,lb,up):
    
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

li=np.array(limit)
population[0]=li[:,0] + np.random.rand(pop, D) * (li[:,1] - li[:,0])  # Inicializa poblacion
population_next[0]=li[:,0] + np.random.rand(pop, D) * (li[:,1] - li[:,0])

#-------------Evaluación población 0------------------------------------------------------------------
for i, xi in enumerate(population[0,:]):  # Evalua objetivos
    solu=pendulum_s(xi,pardyna)
    f_x[0][i], g_x[0][i] =solu[0],solu[1] #function(xi,pardyna)
    #------------------------------------------------------------------------------------------------------

for i in range(0,gen-1):
    f_x_next[i][:]=f_x[i][:]
    population_next[i][:]=population[i][:]
    g_x_next[i][:]=g_x[i][:]
    
    #print ('Generación:',i) 
    selecc=selec(f_x[i,:],g_x[i,:],population[i])
    f_x_s=selecc[0]
    popu_x_s=selecc[1]
    g_x_s=selecc[2]
    
    cross=[]
    if len(f_x_s) % 2 != 0:
        r1 = random.randint(0, len(popu_x_s)-1)
        p1=[]
        p1.append(popu_x_s[r1,0])
        p1.append(popu_x_s[r1,1])
        p1.append(popu_x_s[r1,2])
    lb=np.zeros((len(f_x_s),D))
    up=np.ones((len(f_x_s),D))   
    for j in range(math.floor(len(popu_x_s)/2)):
        
        r1=j
        r2=j
        while r1 == j:
            r1 = random.randint(0, len(popu_x_s)-1)

        while r2 == r1 or r2 == j:
            r2 = random.randint(0, len(popu_x_s)-1)
        p1=[]
        p1.append(popu_x_s[r1,0])
        p1.append(popu_x_s[r1,1])
        p1.append(popu_x_s[r1,2])
        p2=[]
        p2.append(popu_x_s[r2,0])
        p2.append(popu_x_s[r2,1])
        p2.append(popu_x_s[r2,2])
        
        eta=1
        c=crossov(p1,p2,eta,lb[j],up[j])
        cross.append(c[0])
        cross.append(c[1])
    cro=np.array(cross)
    mut=mutPolynomial(cro,1,lb,up)
    f_x_off=np.zeros((len(mut),M))
    g_x_off=np.zeros(len(mut))
   
    for r in range(len(mut)):
        mut[r]=asegurar_limites(mut[r],limit)
        val=pendulum_s(mut[r],pardyna)
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
        # print(f_x_next[i][r])
        # print(g_x_next[i][r])
                

    # Una vez que termina la generacion actualizo x y f_x
    f_x[i+1] = np.copy(f_x_next[i])
    population[i+1] = np.copy(population_next[i])
    g_x[i+1] = np.copy(g_x_next[i])
        
    # print(f_x[i+1])
    # print(g_x[i+1])
        
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
print(a)
print(f_a)

plt.figure(1)
plt.title('Aproximacion al frente de Pareto')

plt.scatter(f_a[:, 0], f_a[:, 1])
#plt.xlim([0,1])

plt.xlabel('f1')
plt.ylabel('f2')
plt.show()
        
    
    
    
    

        
    
    
        

