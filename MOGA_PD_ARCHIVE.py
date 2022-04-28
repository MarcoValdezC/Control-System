# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 21:47:52 2022

@author: marco
"""

import numpy as np
import random
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt

#---------------------Parametros DE-----------------------------#
limit=[[0,8],[0,5],[0,5],[0,5]]       # Limites inferior y superior
pop = 100                    # Tamaño de la población, mayor >= 4
gen =  1000              # Número de generaciones
D= 4                             # Dimensionalidad O número de variables de diseño 
M= 2                              # Numero de objetivos
AMAX = 30                          # Numero maximo de soluciones en el archivo
eta=1

pardyna=[0.5,0.5,1,0.5,1,0.3,0.05,0.02,0.006,0.004]

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
        Minv=inv(M)
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


#------------------------------------------------------------------
def dominates(_a, _b):
    for _j in range(M):               #Recorre el vector J de funciones objetivo
        if _b[_j] < _a[_j]:    
            return False              #Regresa False si a domina b, en este caso seleccionamos b
    return True                       #Regresa Trux si b domina a, en este caso seleccionamos a
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
def selec(f, g, po, D, M):
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
#------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------
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


def moga( limites, poblacion,eta, generaciones,D,M,AMAX,function,pardyna):
    
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
    
    li=np.array(limit)
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
        selecc=selec(f_x[i,:],g_x[i,:],population[i],D,M)
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
        mut=mutPolynomial(cro,eta,lb,up,D)
        f_x_off=np.zeros((len(mut),M))
        g_x_off=np.zeros(len(mut))
        
        for r in range(len(mut)):
            mut[r]=asegurar_limites(mut[r],limit)
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


Hvgapd=np.zeros(30)

for k in range(30):
    print(k)
    var= moga( limit, pop,eta, gen,D,M,AMAX,double_pendulum,pardyna)
    t=var[0]
    x=t[:,0]
    y=t[:,1]
   
    x = np.sort(x)
    y = np.sort(y)[::-1]
    print(x)
    print(y)
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
        Hvgapd[k]=area2
filename="Hvolmogapd.csv" 
myFile=open(filename,'w') 
myFile.write("Hv \n") 
for l in range(len(Hvgapd)): 
    myFile.write(str(Hvgapd[l])+"\n")  
myFile.close()