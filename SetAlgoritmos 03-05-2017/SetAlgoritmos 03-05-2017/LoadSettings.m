%% Carga par�metros de un archivo de inicializaci�n
%% Entrada:
%% Salida:
%   G_MAX: generaci�n m�xima
%   NP: tama�o de poblaci�n
%   CR: factor de cruza
%   F: factor de mutaci�n
%   D: dimensionalidad
%   D_MIN: valores m�nimos de las variables de dise�o
%   D_MAX: valores m�ximos de las variables de dise�o
%   NUM_FO: n�mero de funciones objetivo
%   FO: vector de funciones objetivo
%	NUM_CON: n�mero de restricciones
%   CON: vector de restricciones
%   NS: porcentaje de generaciones antes de utilizar Crowding
%   CW_P: porcentaje del archivo externo utilizado para generar un nuevo
%   individuo
%   DIR: carpeta que contiene los archivos de salida (resultados)
%   RUNS: n�mero de ejecuciones del algoritmo
%   VER: funci�n para generar un nuevo individuo de acuerdo a la versi�n de
%   Evoluci�n Diferencial
function [RUNS,G_MAX,SAVE_G,NP,D,D_MIN,D_MAX]=LoadSettings()
settingsPath = strcat(GetCurrentPath(),'\settings.ini');
PROBLEMS_DIR='problemas';
ALGORITHMS_DIR='algoritmos';

ini = IniConfig();
ini.ReadFile(settingsPath);
sections = ini.GetSections();

G_MAX=ini.GetValues(sections{1},'G_MAX');
NP=ini.GetValues(sections{1},'NP');
%D=ini.GetValues(sections{1},'D');


%D_MIN=ini.GetValues(sections{1},'D_MIN');
%D_MAX=ini.GetValues(sections{1},'D_MAX');

%F_NAME=ini.GetValues(sections{1},'F');
%F=str2func(F_NAME);

%CON=ini.GetValues(sections{1},'CON');
%constraintNames = textscan(CON, '%s', 'delimiter', ' ');
%numCON=size(constraintNames{1});
%NUM_CON=numCON(1);
%CON={NUM_CON};
%for i=1:NUM_CON
%    CON{i}=str2func(constraintNames{1}{i});
%end

global ALG_NAME;
ALG_NAME=ini.GetValues(sections{1},'ALG');
%ALG=str2func(ALG_NAME);

RUNS=ini.GetValues(sections{1},'RUNS');
SAVE_G=ini.GetValues(sections{1},'SAVE_G');

global P_DIR;
P_DIR=ini.GetValues(sections{1},'P_DIR');

%% Agrega carpeta del problema
addpath(sprintf('%s\\%s\\%s',GetCurrentPath(),PROBLEMS_DIR,P_DIR));
%% Agrega carpeta del algoritmo
addpath(sprintf('%s\\%s\\%s',GetCurrentPath(),ALGORITHMS_DIR,ALG_NAME));

%%Limites
Limites;
[~,D]=size(D_MAX);
end