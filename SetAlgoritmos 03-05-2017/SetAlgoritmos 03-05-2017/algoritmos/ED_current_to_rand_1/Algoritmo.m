function sol=Algoritmo(RUN,G_MAX,SAVE_G,NP,D,D_MIN,D_MAX)
%% Debe mantenerse
Parametros;

fprintf('====== Evolucion Diferencial current-to-rand/1 (run %d) ======\n',RUN);
fprintf('*** Parametros ***\n');
fprintf('F = %f\n',F);
fprintf('K = %f\n',K);

F_OFFSET=D+1;
CON_OFFSET=D+2;

pop=zeros(NP,D+2,G_MAX); %Matriz de poblacion (el +2 es por el valor de la funcion objetivo y el contador de violacion de restricciones)

%% Implementacion del algoritmo
%%Inicializa poblacion
g=1;
for i=1:NP
    %%Genera individuo aleatorio
    for j=1:D
        pop(i,j,g)=D_MIN(j)+rand()*(D_MAX(j)-D_MIN(j));
    end
    %%Evalua
    [fit,con]=Evaluar(pop(i,:,g));
    pop(i,F_OFFSET,g)=fit;
    pop(i,CON_OFFSET,g)=con;
end

%%Ciclo evolutivo
while g<G_MAX
    %%Para cada individuo en la generacion actual
    for i=1:NP
        %%Selecciona tres individuos padres diferentes
        r1=randi(NP);
        while r1==i
            r1=randi(NP);
        end
        r2=randi(NP);
        while r2==i || r2==r1
            r2=randi(NP);
        end
        r3=randi(NP);
        while r3==i || r3==r2 || r3==r1
            r3=randi(NP);
        end
        %%Cruza y mutacion (crea un individuo hijo)
        for j=1:D
            pop(i,j,g+1)=pop(i,j,g)+K*(pop(r3,j,g)-pop(i,j,g))+F*(pop(r1,j,g)-pop(r2,j,g));
            
            if pop(i,j,g+1)<=D_MIN(j) || pop(i,j,g+1)>=D_MAX(j)
                pop(i,j,g+1)=D_MIN(j)+rand()*(D_MAX(j)-D_MIN(j));
            end
        end
        
        %%Evalua hijo
        [fit,con]=Evaluar(pop(i,:,g+1));
        pop(i,F_OFFSET,g+1)=fit;
        pop(i,CON_OFFSET,g+1)=con;
        
        %%Seleccion (criterio de Deb)
        if pop(i,CON_OFFSET,g)==0 && pop(i,CON_OFFSET,g+1)==0
            %% Padre e hijo factibles
            %Selecciona al mejor (con base en la funcion objetivo)
            if pop(i,F_OFFSET,g)<pop(i,F_OFFSET,g+1)
                %Pasa padre
                pop(i,:,g+1)=pop(i,:,g);
            else
                %Pasa hijo
                %pop(i,:,g+1)=pop(i,:,g+1);
            end
        elseif pop(i,CON_OFFSET,g) < pop(i,CON_OFFSET,g+1)
            %% Padre viola menos restricciones que hijo
            %Pasa padre
            pop(i,:,g+1)=pop(i,:,g);
        elseif pop(i,CON_OFFSET,g) > pop(i,CON_OFFSET,g+1)
            %% Hijo viola menos restricciones que padre
            %Pasa hijo
            %pop(i,:,g+1)=pop(i,:,g+1);
        else
            %% Padre e hijo violan la misma cantidad de restricciones
            %Selecciona aleatoriamente (con la misma probabilidad)
            if rand()<0.5
                %Pasa padre
                pop(i,:,g+1)=pop(i,:,g);
            else
                %Pasa hijo
                %pop(i,:,g+1)=pop(i,:,g+1);
            end
        end
    end
    g=g+1;
    %%Almacena informacion de la poblacion
    Save(pop,RUN,g,SAVE_G);
end
%%Obtiene la mejor solucion
sol=min(pop(:,:,g));
end
