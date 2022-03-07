function SetAlgoritmos
clc;
close all;
format long g;
%% Inicializa el generador de n�meros aleatorios una vez por instancia de MATLAB
global RNG;

RNG=int16(RNG);
if isempty(RNG)
    RNG=0;
end
if (RNG~=1)    %No funciona del todo pero OK. Cuando no funciona es al poner clear all
    rng('shuffle');
    fprintf('*** N�mero aleatorio inicializado ***\n');
    RNG=1;
end


%% Carga los par�metros del programa
[RUNS,G_MAX,SAVE_G,NP,D, D_MIN,D_MAX]=LoadSettings();

for runNum=1:RUNS
    tic;
    sol=Algoritmo(runNum,G_MAX,SAVE_G,NP,D,D_MIN,D_MAX);
    time=toc;
    fprintf('*** Soluci�n ***\n');
    fprintf('Tiempo:\n');
    fprintf('t = %f s\n',time);
    fprintf('Funcion objetivo:\n');
    fprintf('\tfitness(X) = %f\n',sol(D+1));
    fprintf('Violacion de restricciones:\n');
    fprintf('\tviol = %d\n',sol(D+2));
    fprintf('Variables de dise�o:\n');
    for i=1:D
        fprintf('\tX%d = %16.10f\n',i,sol(i));
    end
    SaveCI(time,RUNS,G_MAX,SAVE_G,NP,D,D_MIN,D_MAX,runNum);
end

end