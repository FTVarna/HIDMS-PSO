% -----------------------------------------------------------------------  %
% The Heterogeneous Improved Dynamic Multi-Swarm PSO (HIDMS-PSO) Algorithm %
%
% Implemented by Fevzi Tugrul Varna - University of Sussex 2020            %
% -------------------------------------------------------------------------%

% Cite as: ----------------------------------------------------------------%
% F. T. Varna and P. Husbands, "HIDMS-PSO: A New Heterogeneous Improved    % 
%  Dynamic Multi-Swarm PSO Algorithm," 2020 IEEE Symposium Series on       %
%  Computational Intelligence (SSCI), 2020, pp. 473-480,                   %
%  doi: 10.1109/SSCI47803.2020.9308313.                                    %
% -----------------------------------------------------------------------  %
%% expected inputs: fhd,fId,n,d,range where fId=function no., n=swarm size, d=dimension, range=lower and upper bounds
%% e.g. HIDMS_PSO(fhd,4,40,30,[-100 100])

function [fmin] = HIDMSPSO(fhd,fId,n,d,range)
if rem(n,4)~=0, error("** Input Error: n/4 must be equal to an even number **"), end
rand('seed',sum(100*clock));
showProgress=true;
cModel=true;                %communication model enable/disable
Fmax=10^4*d;                %maximum number function evaluations
Tmax=Fmax/n;                %maximum number of iterations
LB=range(1);
UB=range(2);
%% Parameters of HIDMS-PSO
w1 = 0.99+(0.2-0.99)*(1./(1+exp(-5*(2*(1:Tmax)/Tmax-1))));      %nonlinear decrease inertia weight - Sigmoid function
c1 = 2.5-(1:Tmax)*2/Tmax;                                       %personal acceleration coefficient
c2 = 0.5+(1:Tmax)*2/Tmax;                                       %social acceleration coefficient          

UPn=4;                            %unit pop size (constant)
U_n=(n/2)/UPn;                    %number of units (constant)
U=reshape(randperm(n/2),U_n,UPn); %units (U_n-by-UPn matrix)
[master,slave1,slave2,slave3] = feval(@(x) x{:}, num2cell([1,2,3,4])); %unit members' codes

%velocity clamp
MaxV=0.15*(UB-LB);
MinV=-MaxV;

%% Initialisation
V=zeros(n,d);           %initial velocities
X=unifrnd(LB,UB,[n,d]); %initial positions
PX=X;                   %initial pbest positions
F=feval(fhd,X',fId);    %function evaluation
PF=F;                   %initial pbest costs
GX=[];                  %gbest solution vector
GF=inf;                 %gbest cost

%update gbest
for i=1:n
    if PF(i)<GF, GF=PF(i); GX=PX(i,:); end
end

%% Main Loop of HIDMS-PSO
for t=1:Tmax

    diversity(t)=mean(F);
    convergence(t)=GF;
    
    %reshuffle units
    if mod(t,10)==0
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
                if ~isempty(find(U(:,master)==i))                  %particle is master
                    [exemplar,exemplar2]=getExemplarMaster(X,PX,U,master,slave1,slave2,slave3,i,U_n,F,PF);
                    V(i,:) = w*V(i,:)+c1(t)*rand([1 d]).*(exemplar - X(i,:)) + c2(t)*rand([1 d]).*(exemplar2 - X(i,:));
                else                                                %particle is slave
                    [exemplar,exemplar2]=getExemplarSlave(X,PX,U,master,slave1,slave2,slave3,i);
                    V(i,:)=w*V(i,:)+c1(t)*rand([1 d]).*(exemplar - X(i,:)) + c2(t)*rand([1 d]).*(exemplar2 - X(i,:));
                end
            else                                                        %velocity update for particles in the homogeneous subpop
                V(i,:)=w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(GX - X(i,:));
            end
        else                                                            %final phase of the search process (exploitation)
            V(i,:)=w*V(i,:)+c1(t)*rand([1 d]).*(PX(i,:) - X(i,:)) + c2(t)*rand([1 d]).*(GX - X(i,:));
        end
        
        V(i,:)=max(V(i,:),MinV); V(i,:) = min(V(i,:),MaxV);           %velocity clamp
        X(i,:)=X(i,:) + V(i,:);                                       %update position
        X(i,:)=max(X(i,:), LB); X(i,:) = min(X(i,:), UB);             %apply lower and upper bound limits
        X(i,:)=Non_uniform_mutation(X(i,:),0.1,t,Tmax,LB,UB);         %apply mutation

        F(i)=feval(fhd,X(i,:)',fId);                    %evaluate fitness function
        if F(i)<PF(i), PF(i)=F(i); PX(i,:)=X(i,:); end  %update pbest
        if PF(i)<GF, GF=PF(i); GX=PX(i,:); end          %update gbest

        %particle communication
        if cModel==true
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
    
    if showProgress==true
        disp(['Iteration '   num2str(t)  ' | Best cost = '  num2str(GF)]);
    end
end
fmin=GF;
end

function [exemplar,exemplar2]=getExemplarMaster(X,PX,U,master,slave1,slave2,slave3,i,U_n,F,PF)
behaviour=randi([1 3]);%randomly selected behaviour
LS=randi([0 1]); %learning strategy
[uId,~] = find(U(:,master)==i);

if LS==0 %inward-oriented learning
    if behaviour==1 %move towards the most dissimilar slave
        sList = U(uId,slave1:slave3);          %get slaves of the master
        similarities = zeros(1,length(sList)); %
        for ii=1:length(sList)                 %calculate similarities between master and slave particles
            %similarities(ii) = mae(PF(i)-PF(sList(ii)));
            similarities(ii) = immse(PF(i),PF(sList(ii))); %immse is used instead of mae and mse for faster performance
        end
        [~,dsId] = max(similarities);               %find the most dissimilar agent
        dsId = sList(dsId);                         %idx of the dissimilar slave
        exemplar=PX(i,:);
        exemplar2=PX(dsId,:);
    elseif behaviour==2 %move towards best slave
        sList = U(uId,slave1:slave3);               %idx of slaves in the unit
        slave_costs = [F(sList(1)) F(sList(2)) F(sList(3))];
        [~,bsId] = min(slave_costs);
        exemplar=PX(i,:);
        exemplar2=PX(bsId,:);
    elseif behaviour==3
        sList = U(uId,slave1:slave3);               %get slaves of the master
        slaves_pos = [X(sList(1),:); PX(sList(2),:); X(sList(3),:);];
        exemplar=PX(i,:);
        exemplar2=mean(slaves_pos);
    end
else %outward-oriented learning
    if behaviour==1 %move towards the avg pos of another unit
        rndU = randi([1,U_n],1,1);                  %select a unit randomly
        sList = U(rndU,slave1:slave3);              %get slaves of a random unit
        uX = [PX(U(rndU,master),:); X(sList(1),:); PX(sList(2),:); X(sList(3),:)]; %all positions of the unit
        exemplar=PX(i,:);
        exemplar2=mean(uX);
    elseif behaviour==2 %move towards the master of another unit
        exemplar=PX(i,:);
        exemplar2=PX(U(randi([1 U_n]),master),:);
    elseif behaviour==3 %move towards avg of self unit and master of another unit
        sList = U(find(U(:,master)==i),slave1:slave3);  %get all slaves of the unit
        sList(4) = U(randi([1 U_n]),master);        %add a master particle from a random unit
        exemplar=mean(X(sList,:));
        exemplar2=PX(U(randi([1 U_n]),master),:);
    end
end

end

function [exemplar,exemplar2]=getExemplarSlave(X,PX,U,master,slave1,slave2,slave3,i)

LS=randi([0 1]); %learning strategy

if LS==0 %inward-oriented learning, move towards the master particle
    [uId,~] = find(U(:,slave1:slave3)==i);
    exemplar=PX(i,:);
    exemplar2=PX(U(uId,master),:);
else     %outward-oriented learning
    [~,sType] = find(U==i);                         %find type/role of the slave
    Slist = U(:,sType);                             %get list of all slaves of the same type
    [selfId,~] = find(Slist==i);
    Slist(selfId) = [];                             %remove self from the list
    rndSlave = randperm(length(Slist));             %shuffle the list
    rndSlave = Slist(rndSlave(1));                  %select the first one
    exemplar=PX(i,:);
    exemplar2=X(rndSlave,:);
end
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