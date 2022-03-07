%%Pueden ser constantes
CR=0.5;
F=0.5;
%%O tambien funciones
%F=@() 0.3+rand()*(0.6-0.3);

%%Cadena indicando condiciones del algoritmo (aunque deberían de ser propias por cada uno)
Cond_Algoritmo=sprintf('Se propone CR=%2.2f y F=%2.2f constante',CR,F);
