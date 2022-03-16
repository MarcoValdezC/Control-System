function moead_de()
clc;
%% MOEA/D-DE parameters
params.cr=0.5;
params.f=0.5;
params.pop_size=100;
params.gen_max=250;
params.niche=6;
params.pm = 1/30;
params.method='tc';

%% Multi-objective problem parameters
mop_params=zdt2_params();
params.d=mop_params.d; % Design variables
params.m=mop_params.m; % Objectives
params.c=mop_params.c; % Constraints
params.ub=mop_params.ub; % Upper bounds
params.lb=mop_params.lb; % Lower bounds

%% Indexes
params.fo_index = params.d + 1; % Objective function indexes
params.con_index = params.fo_index + params.m; % Constraint indexes
params.con_indicator_index = params.con_index + params.c; % Constraint indicator (normalized constraint sum) index

%% Subproblems
ideal_point=ones(1,params.m) * Inf;
pop_new = zeros(params.pop_size, params.con_indicator_index);
neighbours = zeros(params.pop_size, params.niche);
weights = zeros(params.pop_size, params.m);
distance_matrix = zeros(params.pop_size,params.pop_size + 1); % Popsize + subproblem index

%% Initialize subproblems
[weights,neighbours,distance_matrix,pop,ideal_point]=initialize_subproblems(weights,neighbours,distance_matrix,ideal_point,params);


%% Evolutionary cycle
for gen=1:params.gen_max
    %% For each subproblem
    for i = 1:params.pop_size
        %% Generate a new individual (DE/rand/1/bin)
        %% Three randomly selected individuals from the parent niche
        r1 = neighbours(i,randi(params.niche));
        while r1 ==i
            r1 = neighbours(i,randi(params.niche));
        end
        r2 = neighbours(i,randi(params.niche));
        while r2 ==i || r2 == r1
            r2 = neighbours(i,randi(params.niche));
        end
        r3 = neighbours(i,randi(params.niche));
        while r3 ==i || r3 == r2 || r3 == r1
            r3 = neighbours(i,randi(params.niche));
        end
        
        jrand = randi(params.d);
        
        %% Crossover
        for j = 1:params.d
            if rand() < params.cr || j == jrand
                pop_new(i,j) = pop(r1,j) + params.f * (pop(r2,j) - pop(r3,j));
                
                %% Repair the value
                if pop_new(i,j) < params.lb(j)
                    pop_new(i,j) = params.lb(j);
                end
                if pop_new(i,j) > params.ub(j)
                    pop_new(i,j) = params.ub(j);
                end
            else
                pop_new(i,j) = pop(i,j);
            end
        end
        
        %% Gaussian mutation
        for j = 1:params.d
            sigma = (params.ub(j) - params.lb(j)) / 20.0;
            if rand() < params.pm
                pop_new(i,j)= normrnd(pop_new(i,j),sigma);
                
                %% Repair the value
                if pop_new(i,j) < params.lb(j)
                    pop_new(i,j) = params.lb(j);
                end
                if pop_new(i,j) > params.ub(j)
                    pop_new(i,j) = params.ub(j);
                end
            end
        end
        
        
        
        %% Evaluate the new individual
        pop_new(i,:)=evaluate_individual(pop_new(i,:),params);
        
        
        %% Update the ideal point if necessary
        for j = 1:params.m
            if ideal_point(j) > pop_new(i,params.fo_index + j-1)
                ideal_point(j) = pop_new(i,params.fo_index + j-1);
            end
        end
        
        %% Update the neighbours
        for k = 1:params.niche
            new_obj = subobjective(weights(neighbours(i,k),:), pop_new(i,:),ideal_point,params);
            old_obj = subobjective(weights(neighbours(i,k),:), pop(neighbours(i,k),:),ideal_point,params);
            
            %% Compare considering constraints (Deb rules)
            if pop(neighbours(i,k),params.con_indicator_index)==0 && pop_new(i,params.con_indicator_index)==0  %% Two feasible solutions
                if new_obj < old_obj
                    pop(neighbours(i,k),:) = pop_new(i,:);
                end
            else
                if pop(neighbours(i,k),params.con_indicator_index) > pop_new(i,params.con_indicator_index)  %% One of them is infeasible
                    pop(neighbours(i,k),:) = pop_new(i,:);
                else
                    if pop(neighbours(i,k),params.con_indicator_index) == pop_new(i, params.con_indicator_index)  %% Both have the same constraint violations
                        if rand() < 0.5
                            pop(neighbours(i,k),:) = pop_new(i,:);
                        end
                    end
                end
            end
            
        end
    end
end

figure(1);
plot(pop(:,params.fo_index),pop(:,params.fo_index+1),'r*');

end


function [weights,neighbours,distance_matrix,pop,ideal_point]=initialize_subproblems(weights,neighbours,distance_matrix,ideal_point,params)
%% Initialize weights, neighbours and distance matrix
[weights,neighbours,distance_matrix]=initialize_weights(weights,neighbours,distance_matrix,params);
%% Initialize individuals
pop=init_pop(params);
%% Evaluate individuals
for i = 1: params.pop_size
    pop(i,:)=evaluate_individual(pop(i,:), params);
end
%% Ideal point
ideal_point=update_ideal_point(params,ideal_point,pop);
end


function [weights,neighbours,distance_matrix]=initialize_weights(weights,neighbours,distance_matrix,params)
%% Assign weights
for i = 1:params.pop_size
    weights(i,1)= i / (params.pop_size);
    weights(i,2) = 1.0 - weights(i,1);
end
%% Calculate distance matrix

%% Index assignation
for i = 1:params.pop_size
    distance_matrix(i,params.pop_size + 1) = i;
end

for i = 1:params.pop_size
    %% Distance calculation
    for k = i+1:params.pop_size
        w = 0;
        for j = 1:params.m
            w = w + (weights(i,j) - weights(k,j))^2;
        end
        distance_matrix(i,k) = w;
        distance_matrix(k,i) = w;
    end
    
    %% Sort rows with respect to the distances of subproblem i
    distance_matrix=sortrows(distance_matrix, i, 'ascend');
    
    %% Assign the neighbours to the subproblem i
    for k = 1: params.niche
        neighbours(i,k) = distance_matrix(k,params.pop_size + 1);
    end
    
    %% Sort rows with respect to the indexes
    distance_matrix=sortrows(distance_matrix, params.pop_size + 1, 'ascend');
end
end


%% Initialize population randomly in the search space
function pop=init_pop(params)
pop = zeros(params.pop_size, params.con_indicator_index );

for i = 1: params.pop_size
    for j = 1:params.d
        pop(i,j) = params.lb(j) + rand()*(params.ub(j)- params.lb(j));
    end
end
end


%% Evaluate an individual in population
function ind=evaluate_individual(ind,params)
res=zdt2(ind);

%% Copy objectives
for i=1:params.m
    ind(params.fo_index + i - 1)= res(i);
end

%% Copy constraints
ci=0;
for j=1:params.c
    ind(params.con_index + j-1)= res(i+j);
    
    if  res(i+j)~= 0
        ci=ci+1;
    end
end

%% Constraint indicator (sum of constraint violations)
ind(params.con_indicator_index) = ci;
end

function ideal_point = update_ideal_point(params,ideal_point,pop)
for i = 1:params.pop_size
    for j = 1:params.m
        if ideal_point(j) > pop(i,params.fo_index+j-1)
            ideal_point(j) = pop(i,params.fo_index + j-1);
        end
    end
end
end

%% Sub-objective
function r=subobjective(w,s,ideal_point,params)
r = 0;
wf=zeros(params.m,1);

switch params.method
    case 'tc'
        %% Tchebycheff approach
        for j = 1:params.m
            wn = w(j);
            if (wn < 0.00001)
                wn = 0.00001;
            end
            wf(j)= wn * abs(s(params.fo_index + j-1) - ideal_point(j));
        end
        
        r = wf(1);
        for j = 1:params.m
            if r<wf(j)
                r = wf(j);
            end
        end
    case 'ws'
        %% Weighted sum approach
        for j = 1:params.m
            r = r + w(j) * s(fo_index + j-1);
        end
end
end