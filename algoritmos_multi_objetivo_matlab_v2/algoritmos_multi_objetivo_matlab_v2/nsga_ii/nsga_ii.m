function nsga_ii()
clc;
%% NSGA-II parameters
params.pc=0.9;
params.pm=1/30;
params.pop_size=100;
params.gen_max=250;
params.etac=20;
params.etam=20;
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
params.rank_index = params.con_indicator_index + 1; % Rank (non-dominated sorting rank) index
params.crowding_index = params.rank_index + 1; % Crowding distance index

%% Initialize population
pop = init_pop(params);

%% Evaluate population
for i=1:params.pop_size
    pop(i,:)=evaluate_individual(pop(i,:),params);
end

%% Obtains the constraint indicator (normalized constraint sum)
pop=normalize_constraint_error(pop,params);

%% Non-dominated sorting
pop = non_dominated_sorting(pop,params);

%% Evolutionary cycle
for gen=1:params.gen_max
    %% Crossover
    pop=crossover(pop,params);
    %% Obtains the constraint indicator (normalized constraint sum)
    pop=normalize_constraint_error(pop,params);
    %% Non-dominated sorting
    pop = non_dominated_sorting(pop,params);
    %% Replacement
    pop = replacement(pop,params);
end

figure(1);
plot(pop(:,params.fo_index),pop(:,params.fo_index+1),'r*');

end

%% Initialize population randomly in the search space
function pop=init_pop(params)
pop = zeros(params.pop_size, params.crowding_index);

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
for j=1:params.c
    ind(params.con_index + j-1)= res(i+j);
end

end

%% Normalize constraint error
function pop=normalize_constraint_error(pop,params)
con_max = zeros(1,params.c);

for j = 1:params.c
    con_max(j) = max(pop(:,params.con_index+j-1));
    
    if con_max(j) == 0
        con_max(j) = 1;
    end
end

[pop_size,~]=size(pop);
for i = 1:pop_size
    pop(i,params.con_indicator_index) = 0;
    for j = 1:params.c
        pop(i,params.con_indicator_index) = pop(i,params.con_indicator_index)+ pop(i,params.con_index + j -1)/ con_max(j);
    end
end
end

function pop=non_dominated_sorting(pop,params)
feasible= [];
infeasible =[];

% Problem type
all_feasible = 1;
all_infeasible =2;
feasible_and_infeasible =3;

rank = 1;

[pop_size,~]=size(pop);

% Segregating feasible and infeasible solutions, and set rank = 0
for  i = 1:pop_size
    pop(i,params.rank_index) = 0;
    if pop(i,params.con_indicator_index)==0
        feasible = [feasible ; pop(i,:)];
    else
        infeasible = [infeasible ; pop(i,:)];
    end
end

% Classifing the problem
[fsize,~]=size(feasible);
[isize,~]=size(infeasible);

if fsize == pop_size
    problem_type = all_feasible;
else
    if isize == pop_size
        problem_type = all_infeasible;
    else
        problem_type = feasible_and_infeasible;
    end
end

% Handling feasible solutions
if problem_type==all_feasible || problem_type==feasible_and_infeasible
    % Ranking solutions
    ranked_solutions = 0;
    
    while ranked_solutions ~= fsize
        for i = 1:fsize
            if feasible(i,params.rank_index) == 0 % If not is ranked yet, then compare with non ranked solutions
                for j = 1:fsize
                    if i ~= j && ( feasible(j,params.rank_index)==0 || feasible(j,params.rank_index) == rank )% If i,j are not the same solutions and j is not ranked yet or is in the current rank
                        if dominates( feasible(j,:), feasible(i,:), params ) % Check if j dominates i, if do i is a dominated solution
                            break;
                        end
                    end
                end
                
                if j == fsize % Set the corresponding rank
                    feasible(i,params.rank_index) = rank;
                    ranked_solutions=ranked_solutions+1;
                end
            end
        end
        rank=rank+1;
    end
    
    feasible=sortrows(feasible,params.rank_index,'ascend');
    
    % Crowding distance assignment
    % Initialize crowding distance
    for i = 1:fsize
        feasible(i,params.crowding_index) = 0.0;
    end
    
    s = 1;
    
    % Obtaning crowding distance in ranking order
    for r = 1:rank
        sub=[];
        % Get solutions of rank r
        while s <= fsize
            if feasible(s,params.rank_index) == r
                sub=[sub;feasible(s,:)];
                s=s+1;
            else
                break;
            end
        end
        
        [ssize,~]=size(sub);
        
        % Enough information to calculate crowding distance
        if (ssize > 2)
            for j = 1: params.m
                % Sorting respect to fj
                sub=sortrows(sub,params.fo_index + j-1,'descend');
                
                % Set extreme points distance as infinity
                sub(1,params.crowding_index) = Inf;
                sub(ssize,params.crowding_index) = Inf;
                
                % Obtaining the maximum and minimum values of each objective function
                fj_max = sub(1,params.fo_index + j-1);
                fj_min = sub(ssize,params.fo_index + j-1);
                
                % Calculating crowding distance
                for i = 2:ssize - 1
                    sub(i,params.crowding_index)=sub(i,params.crowding_index)+ abs((sub(i - 1,params.fo_index + j-1) - sub(i + 1,params.fo_index + j-1)) / (fj_max - fj_min));
                end
            end
        else
            % Initialize crowding distance
            for i =1:ssize
                sub(i,params.crowding_index) = Inf;
            end
        end
        
        % Replace feasible solutions information with sub
        if ssize>1
            sub=sortrows(sub,params.crowding_index,'descend');
        end
        
        k=1;
        for i = s - ssize:s-1
            feasible(i,:) = sub(k,:);
            k=k+1;
        end
    end
end

% Handling infeasible solutions
if problem_type == all_infeasible || problem_type == feasible_and_infeasible
    sortrows(infeasible,params.con_indicator_index,'ascend');
    
    % Assign next ranks
    for i = 1: isize
        infeasible(i,params.rank_index) = rank;
        infeasible(i,params.crowding_index) = Inf;
        rank=rank+1;
    end
end

% Add feasible and infeasible solutions to x
pop=[feasible;infeasible];
end

% Pareto dominance
function res=dominates(a, b, params)
atLeast = 0;
s = 0;

for i = 1:params.m
    if a(params.fo_index + i-1) <= b(params.fo_index + i-1)
        if a(params.fo_index + i-1)< b(params.fo_index + i-1)
            atLeast = 1;
        end
        s=s+1;
    else
        break;
    end
end

if atLeast == true && s == params.m
    res=1;
else
    res=0;
end
end

%% SBX crossover
function pop=crossover(pop,params)
% Select parents by binary tournament
candidate_indexes_a = [];
candidate_indexes_b = [];
parent_indexes = [];

% Fill candidate vectors
for i = 1:params.pop_size
    candidate_indexes_a=[candidate_indexes_a;i];
    candidate_indexes_b=[candidate_indexes_b;i];
end

% Generate a random permutation
candidate_indexes_a=shuffle(candidate_indexes_a);
candidate_indexes_b=shuffle(candidate_indexes_b);

% Binary comparision of candidates to select parents
for i =1:params.pop_size
    % If candidates have different rank
    if pop(candidate_indexes_a(i),params.rank_index) ~= pop(candidate_indexes_b(i),params.rank_index)
        %Minimum rank individual is selected
        if pop(candidate_indexes_a(i),params.rank_index) < pop(candidate_indexes_b(i),params.rank_index)
            parent_indexes=[parent_indexes;candidate_indexes_a(i)];
        else
            parent_indexes=[parent_indexes;candidate_indexes_b(i)];
        end
    else  % If candidates have the same rank
        % Maximum crowding distance individual is selected
        if pop(candidate_indexes_a(i),params.crowding_index) > pop(candidate_indexes_b(i),params.crowding_index)
            parent_indexes=[parent_indexes;candidate_indexes_a(i)];
        else
            parent_indexes=[parent_indexes;candidate_indexes_b(i)];
        end
    end
end

% Generate offpsrings (SBX crossover and polynomial mutation)
for i = 1:params.pop_size/2
    % Use crossover probability
    if rand() < params.pc
        % Get the i-th couple
        parent_index_1 = parent_indexes(2 * (i-1) + 1);
        parent_index_2 = parent_indexes(2 * (i-1) + 2);
        
        offsprings = sbx(pop(parent_index_1,:), pop(parent_index_2,:), params);
        child_1=offsprings(1,:);
        child_2=offsprings(2,:);
        
        % Mutate offsprings
        child_1=poly_mutation(child_1,params);
        child_2=poly_mutation(child_2,params);
        
        % Evaluate offsprings
        child_1=evaluate_individual(child_1,params);
        child_2=evaluate_individual(child_2,params);
        
        % Add offsprings to the population
        pop=[pop;child_1];
        pop=[pop;child_2];
    end
end
end

%% Random permutation
function v=shuffle(v)
v=v(randperm(length(v)));
end

%% Replacement
function pop=replacement(pop,params)
pop = pop(1:params.pop_size,:);
end

%% SBX
function offspring=sbx(parent1, parent2, params)
EPS = 1.0e-14;
offspring=[parent1;parent2];

for i = 1:params.d
    valueX1 = parent1(i);
    valueX2 = parent2(i);
    if rand() <= 0.5
        if (abs(valueX1 - valueX2) > EPS)
            if (valueX1 < valueX2)
                y1 = valueX1;
                y2 = valueX2;
            else
                y1 = valueX2;
                y2 = valueX1;
            end
            
            lowerBound = params.lb(i);
            upperBound = params.ub(i);
            
            ran = rand();
            beta = 1.0 + (2.0 * (y1 - lowerBound) / (y2 - y1));
            alpha = 2.0 - (beta)^(-(params.etac + 1.0));
            
            if (ran <= (1.0 / alpha))
                betaq = (ran * alpha)^((1.0 / (params.etac + 1.0)));
            else
                betaq = (1.0 / (2.0 - ran * alpha))^(1.0 / (params.etac + 1.0));
            end
            c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1));
            
            beta = 1.0 + (2.0 * (upperBound - y2) / (y2 - y1));
            alpha = 2.0 - (beta)^(-(params.etac + 1.0));
            
            if (ran <= (1.0 / alpha))
                betaq = ((ran * alpha))^((1.0 / (params.etac + 1.0)));
            else
                betaq = (1.0 / (2.0 - ran * alpha))^( 1.0 / (params.etac + 1.0));
            end
            c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1));
            
            if (c1 > params.ub(i))
                c1 = params.ub(i);
            end
            if (c1 < params.lb(i))
                c1 = params.lb(i);
            end
            
            if (c2 > params.ub(i))
                c2 = params.ub(i);
            end
            if (c2 < params.lb(i))
                c2 = params.lb(i);
            end
            
            
            if (rand() <= 0.5)
                offspring(1,i) = c2;
                offspring(2,i) = c1;
            else
                offspring(1,i) = c1;
                offspring(2,i) = c2;
            end
        else
            offspring(1,i) = valueX1;
            offspring(2,i) = valueX2;
        end
    else
        offspring(1,i) = valueX1;
        offspring(2,i) = valueX2;
    end
end

end


%% Polynomial mutation
function ind=poly_mutation(ind,params)

for i = 1:params.d
    if rand() <= params.pm
        y = ind(i);
        yl = params.lb(i);
        yu = params.ub(i);
        if yl == yu
            y = yl;
        else
            delta1 = (y - yl) / (yu - yl);
            delta2 = (yu - y) / (yu - yl);
            rnd = rand();
            mutPow = 1.0 / (params.etam + 1.0);
            if (rnd <= 0.5)
                xy = 1.0 - delta1;
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * ((xy)^( params.etam + 1.0));
                deltaq = (val) ^( mutPow) - 1.0;
            else
                xy = 1.0 - delta2;
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * ((xy)^( params.etam + 1.0));
                deltaq = 1.0 - (val)^(mutPow);
            end
            y = y + deltaq * (yu - yl);
            
            if y > params.ub(i)
                y = params.ub(i);
            end
            if y < params.lb(i)
                y = params.lb(i);
            end
        end
        ind(i) = y;
    end
end
end

