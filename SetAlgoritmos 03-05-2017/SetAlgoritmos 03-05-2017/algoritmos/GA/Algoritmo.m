function sol=Algoritmo(RUN,G_MAX,SAVE_G,NP,D,D_MIN,D_MAX)
%% Debe mantenerse
Parametros;

fprintf('====== Genetic Algorithm (GA) (run %d) ======\n',RUN);
fprintf('*** Parametros ***\n');
fprintf('TS = %f\n',TS);
fprintf('CR = %f\n',CR);
fprintf('F = %f\n',F);

%% Valida que el tamnio de la poblacion sea par
if mod(NP,2) ~=0
    NP=NP-1;
end
F_OFFSET=D+1;
CON_OFFSET=D+2;

x=zeros(NP,D+2,G_MAX); %Matriz de posiciones (el +2 es por el valor de la funcion objetivo y el contador de violacion de restricciones)

%% Implementacion del algoritmo
%% Inicializa cromosomas
g=1;
for i=1:NP
    %% Genera diferentes genes
    for j=1:D
        x(i,j,g)=D_MIN(j)+rand()*(D_MAX(j)-D_MIN(j));
    end
    
    %% Evalua (Funcion objetivo y conteo de violacion de restricciones)
    [fo,res]=Evaluar(x(i,:,g));
    x(i,F_OFFSET,g)=fo;
    x(i,CON_OFFSET,g)=res;
end

%% Ciclo evolutivo

while g<G_MAX
    % Para cada par de individuos
    for i=1:2:NP
        %% Selecciona a dos padres por torneo
        p1=SeleccionPorTorneo(x(:,:,g),NP,TS,F_OFFSET,CON_OFFSET);
        p2=SeleccionPorTorneo(x(:,:,g),NP,TS,F_OFFSET,CON_OFFSET);
        
        %% Cruza
        if rand()<CR
            x(i,:,g+1)=CruzaHeuristica(x(:,:,g),p1,p2,D,F_OFFSET,CON_OFFSET);
            x(i+1,:,g+1)=CruzaHeuristica(x(:,:,g),p1,p2,D,F_OFFSET,CON_OFFSET);
        else
            %% Si no hay cruza, pasan ambos padres
            x(i,:,g+1) = x(p1,:,g);
            x(i+1,:,g+1) = x(p2,:,g);
        end
        
        %% Mutacion
        x(i,:,g+1) = MutacionNoUniforme(g,x(i,:,g+1),F,D,D_MAX,D_MIN,G_MAX);
        x(i+1,:,g+1) = MutacionNoUniforme(g,x(i+1,:,g+1),F,D,D_MAX,D_MIN,G_MAX);
        
        %% Evaluacion
        [fo,res]=Evaluar(x(i,:,g+1));
        x(i,F_OFFSET,g+1)=fo;
        x(i,CON_OFFSET,g+1)=res;
        
        [fo,res]=Evaluar(x(i+1,:,g+1));
        x(i+1,F_OFFSET,g+1)=fo;
        x(i+1,CON_OFFSET,g+1)=res;
        
    end
    
    %% Siguiente generacion
    g=g+1;
    
    %% Almacena informacion de la poblacion
    Save(x,RUN,g,SAVE_G);
end
%% Obtiene la mejor solucion
xgbest=x(1,:,g);
for i=2:NP
    xgbest=CriterioDeDeb(x(i,:,g),xgbest,F_OFFSET,CON_OFFSET);
    
    if isequal( xgbest , x(i,:,g))
        xgbest=x(i,:,g);
    end
end

%% Regresa la mejor solucion
sol=xgbest;
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

function best=SeleccionPorTorneo(x,NP,TS,F_OFFSET,CON_OFFSET)
best = randi(NP);
for i=1:TS-1
    k = randi(NP);
    
    xlbest=CriterioDeDeb(x(best,:),x(k,:),F_OFFSET,CON_OFFSET);
    
    if isequal( xlbest , x(k,:))
        best=k;
    end
end
end

function h=CruzaHeuristica(x,p1,p2,D,F_OFFSET,CON_OFFSET)
alpha =rand();
xlbest=CriterioDeDeb(x(p1,:),x(p2,:),F_OFFSET,CON_OFFSET);

h=zeros(1,D+2);

for j = 1:D
    if isequal( xlbest , x(p1,:))
        h(1,j) = alpha*(x(p1,j) - x(p2,j)) + x(p1,j);
    else
        h(1,j) = alpha*(x(p2,j) - x(p1,j)) + x(p2,j);
    end
end
end


function delta=DeltaFunction(k,y,G_MAX)
alpha = rand();
b = 1;
delta = y*(1 - alpha^((1.0 - k / G_MAX)^b));
end

function m=MutacionNoUniforme(k,h,F,D,D_MAX,D_MIN,G_MAX)
m=zeros(1,D+2);
for j = 1:D
    if (rand()<F)
        if (rand() < 0.5)
            m(1,j) = h(1,j) + DeltaFunction(k, D_MAX(j) - h(1,j), G_MAX);
            
        else
            m(1,j) = h(1,j) - DeltaFunction(k, h(1,j) - D_MIN(j), G_MAX);
        end
    end
end
end
