# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 16:12:11 2022

@author: marco
"""

import numpy as np
import random
import math
import matplotlib.pyplot as plt

pop=100
gen=100
limit=[(0.1,1),(0,5)] 
D = 2                             
M = 2   
AMAX=30
eta=1

# #----------------------------------------------------------------------------------------------------
# #Función Bihn and Korn
# def Bihn(x):
#     f1=4*x[0]**2 + 4*x[1]**2
#     f2=(x[0]-5)**2+(x[1]-5)**2
#     g = 0
#     g += 0 if (x[0] - 5) ** 2 + x[1] ** 2 <= 25 else 1
#     g += 0 if (x[0]- 8) ** 2 + (x[1] + 3) ** 2 >= 7.7 else 1
#     return np.array([f1,f2]),g
# #----------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------


def conex(x):
    f1=x[0]
    f2=(1+x[1])/x[0]
    g = 0
    g += 0 if x[1] + 9 * x[0]>= 6 else 1
    g += 0 if -x[1] + 9 * x[0]>= 1 else 1
    return np.array([f1,f2]), g
    

#--------------------------------------------------------------------------------------------------------

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
#---------------------------------Selección elitista------------------------------------------
def selecga(f, g, po, D, M):
    pop_x_r = np.empty((0, D))
    f_x_r = np.empty((0, M))
    g_x_r = np.empty(0)

    for r, f_x_i in enumerate(f):
        
            
        sol_nd = True
        g_x_i=g[r]
            
        for i2, f_a_2 in enumerate(f):
            g_x_i_2=g[i2]
            if r != i2:
                if dominates(f_a_2, f_x_i):
                    sol_nd = False
                    break
        if sol_nd:
            f_x_r = np.append(f_x_r, [f[r]], axis=0)
            pop_x_r = np.append(pop_x_r, [po[r]], axis=0)
            g_x_r = np.append(g_x_r, [g_x_i], axis=0)
        
 
    #print(g_x_r)
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
def moga( limites, poblacion,eta, generaciones,D,M,AMAX,function):
    
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
        solu=function(xi)
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
        #print(g_x_s)
    
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
            p1=[]
            p1.append(popu_x_s[r1,0])
            p1.append(popu_x_s[r1,1])
            
            p2=[]
            p2.append(popu_x_s[r2,0])
            p2.append(popu_x_s[r2,1])
           
        
          
            c=crossov(p1,p2,eta,lb[j],up[j])
            cross.append(c[0])
            cross.append(c[1])
        cro=np.array(cross)
        mut=mutPolynomial(cro,1,lb,up,D)
        f_x_off=np.zeros((len(mut),M))
        g_x_off=np.zeros(len(mut))
        
        for r in range(len(mut)):
            mut[r]=asegurar_limites(mut[r],limit)
            val=function(mut[r])
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
                #print(g_x_i)
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
    #print(a)
    #print (a[:,1])
    print((f_a))
    # filename="fafill.csv" 
    # myFile=open(filename,'w') 
    # myFile.write("f1, f2 \n") 
    # for l in range(len(f_a)): 
    #     myFile.write(str(f_a[l, 0])+","+str(f_a[l, 1])+"\n") 
    # myFile.close() 
    
    plt.figure(1)
    plt.title('Aproximacion al frente de Pareto')
    plt.scatter(f_a[:, 0], f_a[:, 1])
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()
  
   
    return f_a,a
var= moga( limit, pop,eta, gen,D,M,AMAX,conex)
# Hvmogaps=np.zeros(3)

# for k in range(3):
#     print(k)
    
#     t=var[0]
#     x=t[:,0]
#     y=t[:,1]
   
#     x = np.sort(x)
#     y = np.sort(y)[::-1]
#     x_max = 20
#     y_max = 1
#     yd=0
#     area2=0
#     for i in range(len(x)):
#         if i == 0:  # primer elemento
#             yd=0
            
#             area2=0
#             y_d=y_max-y[i]
#             x_d2=x_max-x[i]
#             area2=x_d2*y_d
#         elif (0<i<len(x)-1):
#             y_d=y[i-1]-y[i]
#             x_d2=x_max-x[i]
#             area2=area2+(y_d*x_d2)
#         elif i == len(x)-1:  # ultimo elemento
#             y_d=y[i-1]-y[i]
#             x_d3=x_max-x[i-1]
#             area2=area2+(y_d*x_d3)
#             print('Hipervolumen:')
#             print( area2)
#         Hvmogaps[k]=area2
    
# filename="Hvolmogaps.csv" 
# myFile=open(filename,'w') 
# myFile.write("Hv \n") 
# for l in range(len(Hvmogaps)): 
#     myFile.write(str(Hvmogaps[l])+"\n")  
# myFile.close()
