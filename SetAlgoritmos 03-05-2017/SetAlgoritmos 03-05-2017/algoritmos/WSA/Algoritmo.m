function sol=Algoritmo(RUN,G_MAX,SAVE_G,W,D,D_MIN,D_MAX)
%% Debe mantenerse
Parametros;

fprintf('====== Wolf Search Algorithm (WSA) (run %d) ======\n',RUN);
fprintf('*** Parametros ***\n');
fprintf('V = %f\n',V);
fprintf('S = %f\n',S);
fprintf('Pa = %f\n',Pa);
fprintf('Alpha = %f\n',Alpha);

F_OFFSET=D+1;
CON_OFFSET=D+2;

dist=zeros(W,1); % Vector de distancias
x=zeros(W,D+2,G_MAX); %Matriz de posiciones (el +2 es por el valor de la funcion objetivo y el contador de violacion de restricciones)

%% Implementacion del algoritmo
%% Inicializa posiciones y velocidades
g=1;
for i=1:W
    %% Genera posiciones aleatorias
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
    w=x(:,:,g);
    %% Para cada lobo
    for i=1:W
        %% buscaComidaInicialmente()
        % El lobo busca en su campo v mirando aleatoriamente
        % El campo v está dentro del radio v*alpha
        wtmp=w(i,:);
        for k=1:D
            wtmp(1,k)=wtmp(1,k)+0.2*(-1+2*rand)*V;
        end
        % Evalua nueva posicion
        [fo,res]=Evaluar(wtmp);
        wtmp(1,F_OFFSET)=fo;
        wtmp(1,CON_OFFSET)=res;
        % Si el lobo encuentra una mejor posición en su campo v
        % se mueve hacia dicho lugar (con base en la FO)
        wlbest=CriterioDeDeb(wtmp,w(i,:),F_OFFSET,CON_OFFSET);
        if isequal(wlbest,wtmp)
            w(i,:)=wtmp;
        end
        
        %% hayLobosCerca()
        % Calcula la distancia del lobo hacia los demás lobos
        % (distancia euclidiana)
        for j=1:W
            dist(j)=norm(w(i,:)-w(j,:));
        end
        
        % El lobo busca si hay comPañeros dentro de su radio v*alpha
        % si hay alguno y tiene mejor posición se mueve hacia el (si hay
        % varios se mueve hacia el de mejor posición)
        satisfiedindex=find(dist>0&dist<V*Alpha); % Índices que satisfacen la condición (dist<dist*alpha)
        
        if satisfiedindex>0 % Si hay alguno
            %% caminaHaciaElMejor()
            satisfiedfitness=w(satisfiedindex,F_OFFSET); % Obtiene sus valores de FO
            [~,bestsatisfiedindex]=min(satisfiedfitness); % Obtiene el mejor
            
            % Saca el mejor índice
            localbestindex=satisfiedindex(bestsatisfiedindex);
            bestw=w(localbestindex,:);
            
            % Obtiene la distancia Euclidiana hacia la mejor solución
            sum=0;
            for j=1:D
                sum=sum+(bestw(j)-w(i,j))^2;
            end
            r=sqrt(sum);
            
            % Utiliza función de incentivo (Para guiar la búsqueda hacia la comida)
            Beta0=1;
            Beta1=Beta0*exp(-r.^2);
            wtmp=w(i,:).*(1-Beta1)+bestw.*Beta1+Alpha.*(-1+2*rand);
            % Evalua nueva posicion
            [fo,res]=Evaluar(wtmp);
            wtmp(1,F_OFFSET)=fo;
            wtmp(1,CON_OFFSET)=res;
            
            % Si la nueva posición lo acerca a la comida se mueve
            wlbest=CriterioDeDeb(wtmp,w(i,:),F_OFFSET,CON_OFFSET);
            if isequal(wlbest,wtmp)
                w(i,:)=wtmp;
            end
        else
            %% buscaComidaPasivamente()
            wtmp=w(i,:);
            for k=1:D
                wtmp(k)=wtmp(k)+0.2*(-1+2*rand)*V;
            end
            % Evalua nueva posicion
            [fo,res]=Evaluar(wtmp);
            wtmp(1,F_OFFSET)=fo;
            wtmp(1,CON_OFFSET)=res;
            %Si la nueva posición es mejor, muevete a ella
            wlbest=CriterioDeDeb(wtmp,w(i,:),F_OFFSET,CON_OFFSET);
            if isequal(wlbest,wtmp)
                w(i,:)=wtmp;
            end
        end
        
        %% escapa()
        % Escapa dada la probabilidad Pa
        if rand> Pa
            %Genera una nueva posición aleatoria que es lejana en el radio
            %de v*s
            wtmp=w(i,:)+randn(size(w(i,:)))*V*S;
            % Evalua nueva posicion
            [fo,res]=Evaluar(wtmp);
            wtmp(1,F_OFFSET)=fo;
            wtmp(1,CON_OFFSET)=res;
            %Si la nueva posición es mejor, muevete a ella
            wlbest=CriterioDeDeb(wtmp,w(i,:),F_OFFSET,CON_OFFSET);
            if isequal(wlbest,wtmp)
                w(i,:)=wtmp;
            end
        end
    end
    
    x(:,:,g+1)=w;
    
    %% Obtiene a la mejor de la poblacion
    xgbest=x(1,:,g+1);
    for i=2:W
        xgbest=CriterioDeDeb(x(i,:,g+1),xgbest,F_OFFSET,CON_OFFSET);

        if isequal( xgbest , x(i,:,g+1))
            xgbest=x(i,:,g+1);
        end
    end
    
    %% Incrementa generacion
    g=g+1;
    
    %% Almacena informacion de la poblacion
    Save(x,RUN,g,SAVE_G);
end
%% Obtiene la mejor solucion
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
