function sol=Algoritmo(RUN,G_MAX,SAVE_G,NP,D,D_MIN,D_MAX)
%% Debe mantenerse
Parametros;

fprintf('====== Particle Swarm Optimization (PSO) (run %d) ======\n',RUN);
fprintf('*** Parametros ***\n');
fprintf('C1 = %f\n',C1);
fprintf('C2 = %f\n',C2);
fprintf('Vmax = %f\n',Vmax);
fprintf('Vmin = %f\n',Vmin);

F_OFFSET=D+1;
CON_OFFSET=D+2;

x=zeros(NP,D+2,G_MAX); %Matriz de posiciones (el +2 es por el valor de la funcion objetivo y el contador de violacion de restricciones)
x_best=zeros(NP,D+2); %Matriz de mejores posiciones (locales)
x_best_swarm=zeros(1,D+2); %Mejor posicion (global)

x_dot=zeros(NP,D); %Matriz de velocidades

%% Implementacion del algoritmo
%% Inicializa posiciones y velocidades
g=1;
for i=1:NP
    %% Genera posiciones aleatorias
    for j=1:D
        x(i,j,g)=D_MIN(j)+rand()*(D_MAX(j)-D_MIN(j));
    end
    %% Inicializa velocidades en cero
    for j=1:D
        x_dot(i,j)=0;
    end
    
    %% Evalua (Funcion objetivo y conteo de violacion de restricciones)
    [fo,res]=Evaluar(x(i,:,g));
    x(i,F_OFFSET,g)=fo;
    x(i,CON_OFFSET,g)=res;
    
    %% Inicializa mejor local
    x_best(i,:)=x(i,:,g);
end

%% Inicializa al mejor global
x_best_swarm(1,:)=x_best(1,:);
for i=2:NP
    x_best_swarm(1,:)=CriterioDeDeb(x_best(i,:),x_best_swarm(1,:),F_OFFSET,CON_OFFSET);
end

%% Ciclo evolutivo
while g<G_MAX
    %% Actualiza factor de velocidad (linealmente)
    w=Vmax-(g / G_MAX)*(Vmax-Vmin);
    %% Para cada individuo en la generacion actual
    for i=1:NP
        %% Actualiza velocidad
        B1=rand();
        B2=rand();
        for j=1:D
            x_dot(i,j) = w*x_dot(i,j) + B1*C1*(x_best(i,j) - x(i,j,g)) + B2*C2*(x_best_swarm(1,j) - x(i,j,g));
        end
        
        %% Actualiza posicion
        for j=1:D
            x(i,j,g+1) = x(i,j,g) + x_dot(i,j);
        end
        
        %% Evalua nueva posicion
        [fo,res]=Evaluar(x(i,:,g+1));
        x(i,F_OFFSET,g+1)=fo;
        x(i,CON_OFFSET,g+1)=res;
        
        %% Actaualiza mejor local (criterio de Deb)
        x_best(i,:)=CriterioDeDeb(x_best(i,:),x(i,:,g+1),F_OFFSET,CON_OFFSET);
    end
    
    %% Encuentra al mejor global
    x_best_swarm(1,:)=x_best(1,:);
    for i=2:NP
        x_best_swarm(1,:)=CriterioDeDeb(x_best(i,:),x_best_swarm(1,:),F_OFFSET,CON_OFFSET);
    end
    
    g=g+1;
    
    %% Almacena informacion de la poblacion
    Save(x,RUN,g,SAVE_G);
end
%% Obtiene la mejor solucion
sol=x_best_swarm;
end

function sbest=CriterioDeDeb(s1,s2,F_OFFSET,CON_OFFSET)
%% Seleccion (criterio de Deb)
if s1(CON_OFFSET)==0 && s2(CON_OFFSET)==0
    %% Ambos factibles
    %Selecciona al mejor (con base en la funcion objetivo)
    if s1(F_OFFSET)<s2(F_OFFSET)
        %Pasa s1
        sbest=s1;
    else
        %Pasa s2
        sbest=s2;
    end
elseif s1(CON_OFFSET) < s2(CON_OFFSET)
    %% s1 viola menos restricciones que s2
    %Pasa s1
    sbest=s1;
elseif s1(CON_OFFSET) > s2(CON_OFFSET)
    %% s2 viola menos restricciones que s1
    %Pasa s2
    sbest=s2;
else
    %% Ambos violan la misma cantidad de restricciones
    %Selecciona aleatoriamente (con la misma probabilidad)
    if rand()<0.5
        %Pasa s1
        sbest=s1;
    else
        %Pasa s2
        sbest=s2;
    end
end
end
