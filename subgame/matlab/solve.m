clear all
clc
load utility.mat

U = -1*defender_utility;
C=length(U);

f = [-1 zeros(1, C)];       % objective func only uses extra column
A = [ones(C, 1) U];         % prepend extra column
b = zeros(1, C)';           % each row becomes <= 0
Aeq = [0 ones(1, C)];       % sum ignores extra column
beq = 1;                    % sum equals 1
lb = [-1000000; zeros(C, 1)];  % prob min is 0
ub = [1000000; ones(C, 1)];    % prob max is 1


tic; % start timer
p = linprog(f, A, b, Aeq, beq, lb, ub) % solve for defender
toc; % stop timer

defender_util_expected = sum(defender_utility * p(2:end))


U = -1*attacker_utility;
f = [-1 zeros(1, C)];        % objective func only uses extra column
A = [ones(C, 1) U];       % prepend extra column
b = zeros(1, C)';           % each row becomes <= 0
Aeq = [0 ones(1, C)];       % sum ignores extra column
beq = 1;                    % sum equals 1
lb = [-1000; zeros(C, 1)];  % prob min is 0
ub = [1000; ones(C, 1)];    % prob max is 1

tic; % start timer
q = linprog(f, A, b, Aeq, beq, lb, ub) % solve for defender
toc; % stop timer


attacker_util_expected = sum(attacker_utility * q(2:end))