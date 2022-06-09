% clear all
clc

% numAttacks=3;
% U = rand(numAttacks, numAttacks);

f = [-1 zeros(1, numAttacks)]   % objective func only uses extra column
A = [ones(numAttacks, 1) U]    % prepend extra column
b = zeros(1, numAttacks)'      % each row becomes <= 0
Aeq = [0 ones(1, numAttacks)]  % sum ignores extra column
beq = 1;                       % sum equals 1
lb = [-1000; zeros(numAttacks, 1)]; % prob min is 0
ub = [1000; ones(numAttacks, 1)];   % prob max is 1
tic; % start timer
p = linprog(f, A, b, Aeq, beq, lb, ub) % solve
toc; % stop timer

util = sum(U * p(2:end))