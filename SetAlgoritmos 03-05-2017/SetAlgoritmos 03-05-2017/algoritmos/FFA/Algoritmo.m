function sol=Algoritmo(RUN,G_MAX,SAVE_G,NP,D,D_MIN,D_MAX)
%% Debe mantenerse
Parametros;

fprintf('====== Firefly Algorithm (FFA) (run %d) ======\n',RUN);
fprintf('*** Parametros ***\n');
fprintf('Alpha = %f\n',Alpha);
fprintf('Betamin = %f\n',Betamin);
fprintf('Gamma = %f\n',Gamma);
Beta0=1.0;

F_OFFSET=D+1;
CON_OFFSET=D+2;

x=zeros(NP,D+2,G_MAX); %Matriz de posiciones (el +2 es por el valor de la funcion objetivo y el contador de violacion de restricciones)
xgbest=zeros(1,D+2);
%% Implementacion del algoritmo
%% Inicializa posiciones
g=1;
for i=1:NP
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
Alphad=Alpha;
while g<G_MAX
    %% Decrementa alpha
    Alphad=AlphaNew(Alphad, G_MAX);
   
    %% Mueve luciernagas
    x_new=x(:,:,g);
    % Mueve i hacia j
    for i=1:NP
        for j=1:NP
            %% Calcula el parametro de atraccion
            r=0;
            for k=1:D
                r=r+(x_new(i,k) - x_new(j,k))^2;
            end
            r=sqrt(r);
            
            %% Genera nuevas posiciones
            % Si es mas atractiva y luminosa
            xbest=CriterioDeDeb(x_new(i,:),x_new(j,:),F_OFFSET,CON_OFFSET);
            if isequal(xbest,x_new(j,:))
                beta = (Beta0 - Betamin)*exp(-Gamma*r^2)+Betamin;
                
                for k=1:D
                    x_new(i,k)=x_new(i,k)*(1-beta) +...
                        x_new(j,k)*beta + ...
                        Alphad*(rand() - 0.5)*abs(D_MAX(k)-D_MIN(k));
                    
                    if x_new(i,k) < D_MIN(k)
                        x_new(i,k)=D_MIN(k);
                    end

                    if x_new(i,k) > D_MAX(k)
                        x_new(i,k)=D_MAX(k);
                    end
                end
            end
        end
    end

    x(:,:,g+1) = x_new;
    
    %% Evalua las nuevas posiciones
    for i=1:NP
        %% Evalua (Funcion objetivo y conteo de violacion de restricciones)
        [fo,res]=Evaluar(x(i,:,g+1));
        x(i,F_OFFSET,g+1)=fo;
        x(i,CON_OFFSET,g+1)=res;
    end
    
    %% Ordena con base en luminosidad (menor valor de FO y de violacion de restricciones)
    for i=2:NP
        for j=1:NP-i
            %% Burbuja
            xlbest=CriterioDeDeb(x(j,:,g+1),x(j+1,:,g+1),F_OFFSET,CON_OFFSET);
            
            if isequal(xlbest,x(j,:,g+1))
                xaux = x(j,:,g+1);
                x(j,:,g+1) = x(j+1,:,g+1);
                x(j+1,:,g+1)=xaux;
            end
        end
    end
    
    %% Obten a la mejor luciernaga
    xgbest=x(1,:,g+1);
    for i=2:NP
        xgbest=CriterioDeDeb(x(i,:,g+1),xgbest,F_OFFSET,CON_OFFSET);

        if isequal( xgbest , x(i,:,g+1))
            xgbest=x(i,:,g+1);
        end
    end
    
    %% Siguiente generacion
    g=g+1;
    
    %% Almacena informacion de la poblacion
    Save(x,RUN,g,SAVE_G);
end
%% Obtiene la mejor solucion
sol=xgbest;
end

function anew=AlphaNew(alphad, g_max)
delta = 1.0 - (10^(-4) / 0.9) ^ (1.0 / g_max);
anew = (1.0 - delta)*alphad;
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
