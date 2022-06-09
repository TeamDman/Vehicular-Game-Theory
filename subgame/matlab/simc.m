clear all
clc
%Simulation parameters
y1=[];y2=[];y3=[];y4=[];time=[];
delta=0.9;%detection rate
for C=[100 200 300 400 500]
    %Utility computing
    W=0.5+0.5*rand(1,C);%weight of each RSU in the decision
    cost=0.2*rand(1,C);%monitoring cost for each RSU (depends on infomation volume)
    D=rand(1,C);%damage of attack w.r.t to each RSU
    U=zeros(C); % system utility matrix
    for i=1:C
        for j=1:C
            if i==j
                U(i,j) = -1*(1-delta)*D(i)*W(i);
            else
                U(i,j) = -1*W(j)*D(j);
            end
        end
    end
    
    %Optimization problem (system)
    f=[zeros(1,C)];
    lb=[zeros(C,1)];
    ub=[ones(C,1)];
    Aeq=[ones(1,C)];
    beq=1;
    A=[-1*U'];
    b=zeros(1,C);
    tic
    [p,P] = linprog(-1*f, A, b, Aeq, beq, lb, ub);
    time(end+1)=toc;
    
    %Optimization problem (attacker)
    f=[zeros(1,C)];
    lb=[zeros(C,1)];
    ub=[ones(C,1)];
    Aeq=[ones(1,C)];
    beq=1;
    A=[U];
    b=zeros(1,C);
    [q,Q] = linprog(f, A, b, Aeq, beq, lb, ub);
    
    pp=[];qq=[];K=[];
    for n=1:C
        pp=[pp p(1:end)];
    end
    for n=1:C
        qq=[qq q(1:end)];
    end
    for n=1:C
        K=[K ones(C,1)/C];
    end
            
    y1(end+1)=max(sum(U'.*qq));
    y2(end+1)=max(sum(U'.*K));
    % %optimal utility value
    y3(end+1)=min(sum(U.*pp));
    % %fair-based utility value
    % % y2=max(U*((1/C)*ones(C,1).*q(2:end)));
    % %y1=max(U*(p(2:end).*ones(C,1).*q(2:end)));
    y4(end+1)=min(sum(U.*K));
    
end
figure(1)
hold on
grid on
plot(100:100:500,y1,'b','marker','o');
plot(100:100:500,y2,'r','marker','d');
XTick = 100:100:500;
set(gca,'xtick',XTick)
xticklabels({'100','200','300','400','500'});
xlabel('Number of RSUs');
ylabel('Maximum System Utility');
legend('Optimal Attack Strategy','Fair Attack Strategy');


figure(2)
hold on
grid on
plot(100:100:500,y3,'b','marker','o');
plot(100:100:500,y4,'r','marker','d');
xlabel('Number of RSUs');
ylabel('Minimum System Utility');
XTick = 100:100:500;
set(gca,'xtick',XTick);
xticklabels({'100','200','300','400','500'});
legend('Optimal Defense Strategy','Fair Defense Strategy');

time = [0.19201000000000,0.1663152000000000,0.189048500000000,0.122033300000000,0.192196700000000];
figure(3)
grid on
plot(100:100:500,time,'g','marker','*','linewidth',2);
xlabel('Number of RSUs');
ylabel('Execution Time (s)');
XTick = 100:100:500;
set(gca,'xtick',XTick);
xticklabels({'100','200','300','400','500'});
grid on
ylim([0 0.3]);


