function params=zdt2_params()
%% Multi-objective problem parameters
params.d=30; % Design variables
params.m=2; % Objectives
params.c=0; % Constraints
params.ub=ones(params.d,1); % Upper bounds
params.lb=zeros(params.d,1); % Lower bounds