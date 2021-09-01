% -----------------------------------------------------------------------  %
% The Heterogeneous Improved Dynamic Multi-Swarm PSO (HIDMS-PSO) Algorithm %
%
% Programmed by Fevzi Tugrul Varna - University of Sussex 2020            %
% -------------------------------------------------------------------------%

% Reference: --------------------------------------------------------------%
% F. T. Varna and P. Husbands, "HIDMS-PSO: A New Heterogeneous Improved    % 
%  Dynamic Multi-Swarm PSO Algorithm," 2020 IEEE Symposium Series on       %
%  Computational Intelligence (SSCI), 2020, pp. 473-480,                   %
%  doi: 10.1109/SSCI47803.2020.9308313.                                    %
% -----------------------------------------------------------------------  %
% expected inputs: fId,n,LB,UB,d where fId=function no., n=population, LB,UB=lower and upper bounds, d=dimension
% e.g. HIDMS_PSO(1,40,-100,100,30)

function [fmin] = HIDMS_PSO(fId,n,LB,UB,d)
if rem(n,4)~=0, error("** Input Error: Population must be divisible by 4 **"), end
rand('seed',sum(100*clock));
COM=false;
Fmax=10^4*d;                %maximum number function evaluations
Tmax=Fmax/n;                %maximum number of iterations
ShowProgress=false;
fhd=str2func('cec17_func'); %only use if ran on the CEC'17 benchmark test suite

%% Parameters of HIDMS-PSO
w1 = 0.99+(0.2-0.99)*(1./(1+exp(-5*(2*(1:Tmax)/Tmax-1))));      %nonlinear decrease inertia weight - Sigmoid function
c1 = 2.5-(1:Tmax)*2/Tmax;                                       %cognitive acceleration coefficient
c2 = 0.5+(1:Tmax)*2/Tmax;                                       %social acceleration coefficient

%alpha parameter - determines units' reshape interval
alpha_min=Tmax*0.01;
alpha_max=Tmax*0.1;
alpha=alpha_max;  %initial alpha value

UPn=4;                            %unit pop size (constant)
U_n=(n/2)/UPn;                    %number of units (constant)
U=reshape(randperm(n/2),U_n,UPn); %units (U_n-by-UPn matrix)
[master,slave1,slave2,slave3] = feval(@(x) x{:}, num2cell([1,2,3,4])); %unit members' codes (constant)

%Velocity clamp
MaxV=0.08*(UB-LB);
MinV=-MaxV;

%% Initialisation
V=zeros(n,d);           %initial velocities
X=unifrnd(LB,UB,[n,d]); %initial positions
PX=X;                   %initial pbest positions
F=feval(fhd,X',fId);    %function evaluation - for the CEC'17 test suite - alter according to the objective function being optimised
%F=benchmark_func(X,fId); %function evaluation using the CEC'05 test suite
PF=F;                   %initial pbest costs
GX=[];                  %gbest solution vector
GF=inf;                 %gbest cost

%update gbest
for i=1:n
    if PF(i)<GF, GF=PF(i); GX=PX(i,:); end
end

%% Main Loop of HIDMS-PSO
for t=1:Tmax
    %reshape units
    if mod(t,alpha)==0
        [~,idx]=sort(rand(U_n,UPn));
        U=U(sub2ind([U_n,UPn],idx,ones(U_n,1)*(1:UPn)));
    end
    
    for i=1:n
        if F(i) >= mean(F)
            w = w1(t) + 0.15;
            if w>0.99,  w = 0.99;end
        else
            w = w1(t) - 0.15;
            if w<0.20,  w = 0.20;end
        end
        
        if t<=Tmax*0.9
            if ~isempty(find(U==i))                                %if agent belongs to the heterogeneous subpop
                if randi([0 1])==0                                 %inward-oriented movement
                    if ~isempty(find(U(:,master)==i))              %agent is master
                        behaviour = randi([1 3]);
                        [uId,~] = find(U(:,master)==i);            %unit id of the ith (master) particle
                        if behaviour==1                            %move towards the most dissimilar slave
                            sList = U(uId,slave1:slave3);          %get slaves of the master
                            similarities = zeros(1,length(sList)); %
                            for ii=1:length(sList)                 %calculate similarities between master and slave particles
                                %similarities(ii) = mae(PF(i)-PF(sList(ii)));
                                similarities(ii) = immse(PF(i),PF(sList(ii))); %immse is used instead of mae and mse for faster performance
                            end
                            [~,dsId] = max(similarities);               %find the most dissimilar agent
                            dsId = sList(dsId);                         %idx of the dissimilar slave
                            V(i,:)=w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(PX(dsId,:) - X(i,:));
                        elseif behaviour == 2                           %move towards the best slave
                            sList = U(uId,slave1:slave3);               %idx of slaves in the unit
                            slave_costs = [F(sList(1)) F(sList(2)) F(sList(3))];
                            [~,bsId] = min(slave_costs);                %best slave's idx
                            V(i,:)=w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(PX(sList(bsId),:) - X(i,:));
                        elseif behaviour == 3                           %move towards average of slaves
                            sList = U(uId,slave1:slave3);               %get slaves of the master
                            slaves_pos = [X(sList(1),:); PX(sList(2),:); X(sList(3),:);];
                            V(i,:)=w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(mean(slaves_pos) - X(i,:));
                        end
                    else                                                %agent is slave, move towards the master particle
                        [uId,~] = find(U(:,slave1:slave3)==i);          %find the unit particle belongs to
                        V(i,:)=w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(PX(U(uId,master),:) - X(i,:));
                    end
                else	%outward-oriented movement
                    if ~isempty(find(U(:,master)==i))                   %agent is master
                        behaviour = randi([1 3]);                       %randomly selected behaviour
                        if behaviour==1                                 %if 1, move towards the avg pos of another unit
                            rndU = randi([1,U_n],1,1);                  %select a unit randomly
                            sList = U(rndU,slave1:slave3);              %get slaves of a random unit
                            uX = [PX(U(rndU,master),:); X(sList(1),:); PX(sList(2),:); X(sList(3),:)]; %all positions of the unit
                            V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(mean(uX) - X(i,:));
                        elseif behaviour==2                             %move towards the master of another unit
                            V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(PX(U(randi([1 U_n]),master),:) - X(i,:));
                        elseif behaviour==3                             %move towards avg of self unit and master of another unit
                            sList = U(find(U(:,master)==i),slave1:slave3);  %get all slaves of the unit
                            sList(4) = U(randi([1 U_n]),master);        %add a master particle from a random unit
                            V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(mean(X(sList,:)) - X(i,:)) + c2(t)*rand([1 d]).*(PX(U(randi([1 U_n]),master),:) - X(i,:));
                        end
                    else                                                %agent is slave
                        [~,sType] = find(U==i);                         %find self slave type
                        Slist = U(:,sType);                             %get list of all slaves of the same type
                        [selfId,~] = find(Slist==i);
                        Slist(selfId) = [];                             %remove self from the list
                        rndSlave = randperm(length(Slist));             %shuffle the list
                        rndSlave = Slist(rndSlave(1));                  %select the first one
                        V(i,:)=w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(X(rndSlave,:) - X(i,:));
                    end
                end
            else                                                        %velocity update for particles in the homogeneous subpop  
                V(i,:)=w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(GX - X(i,:));
            end
        else                                                            %final phase of the search process (exploitation)
            V(i,:)=w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(GX - X(i,:));
        end
        
        V(i,:) = max(V(i,:),MinV); V(i,:) = min(V(i,:),MaxV);           %velocity clamp
        X(i,:) = X(i,:) + V(i,:);                                       %update position
        X(i,:) = max(X(i,:), LB); X(i,:) = min(X(i,:), UB);             %apply lower and upper bound limits
        X(i,:) = Non_uniform_mutation(X(i,:),0.1,t,Tmax,LB,UB);         %apply mutation
        
        %particle communication
        if COM==true
            if ~isempty(find(U==i))                                     %if particle belongs to the heterogeneous subpop
                [~,sIdx] = find(U==i);                                  %find the slave type of the ith particle
                if sIdx == slave1 || sIdx == slave2 || sIdx == slave3   %if ith agent is a slave
                    sList = U(:,sIdx);                                  %pool of same type slaves from all units
                    rndSlave = randperm(length(sList),1);               %select a random slave from the pool
                    %positional info exchange between the ith particle and a random slave of the same type
                    if PF(i)<PF(sList(rndSlave)), PF(sList(rndSlave))=PF(i); PX(sList(rndSlave),:)=PX(i,:);
                    else, PF(i)=PF(sList(rndSlave)); PX(i,:)=PX(sList(rndSlave),:);
                    end
                end
            end
        end
    end
    
    %function evaluation
    F=feval(fhd,X',fId); %CEC'17
    %F=benchmark_func(X,fId); %CEC'05
    
    for j=1:n
        if F(j)<PF(j), PF(j)=F(j); PX(j,:)=X(j,:); end  %update pbests
        if PF(j)<GF, GF=PF(j); GX=PX(j,:); end          %update gbest
    end
    
    alpha=round(alpha_max-(alpha_max-alpha_min)*t/Tmax);
    
    if ShowProgress==true
        disp(['Iteration '   num2str(t)  ' | Best cost = '  num2str(GF)]);
    end
end
fmin=GF;
end

%% Nonuniform Mutation Operator
function [y] = Non_uniform_mutation(x,p,t,Tmax,LB,UB)
b=5;                %system parameter - 2~5
[m,n]=size(x);
y = x;
for i=1:m
    for j=1:n
        if rand<p
            D = diag(rand(1,n));
            if round(rand)==0
                y(i,j) = x(i,j)+D(j,j)*(UB-x(i,j))*(1-t/Tmax)^b;
            else
                y(i,j) = x(i,j)-D(j,j)*(x(i,j) - LB)*(1-t/Tmax)^b;
            end
        end
    end
end
end
