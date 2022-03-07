function SetAlgoritmos
clc;
close all;
format long g;
%% Inicializa el generador de números aleatorios una vez por instancia de MATLAB
global RNG;

RNG=int16(RNG);
if isempty(RNG)
    RNG=0;
end
if (RNG~=1)    %No funciona del todo pero OK. Cuando no funciona es al poner clear all
    rng('shuffle');
    fprintf('*** Número aleatorio inicializado ***\n');
    RNG=1;
end


%% Carga los parámetros del programa
[RUNS,G_MAX,SAVE_G,NP,D, D_MIN,D_MAX]=LoadSettings();

for runNum=1:RUNS
    tic;
    sol=Algoritmo(runNum,G_MAX,SAVE_G,NP,D,D_MIN,D_MAX);
    time=toc;
    fprintf('*** Solución ***\n');
    fprintf('Tiempo:\n');
    fprintf('t = %f s\n',time);
    fprintf('Funcion objetivo:\n');
    fprintf('\tfitness(X) = %f\n',sol(D+1));
    fprintf('Violacion de restricciones:\n');
    fprintf('\tviol = %d\n',sol(D+2));
    fprintf('Variables de diseño:\n');
    for i=1:D
        fprintf('\tX%d = %16.10f\n',i,sol(i));
    end
    SaveCI(time,RUNS,G_MAX,SAVE_G,NP,D,D_MIN,D_MAX,runNum);
end

end