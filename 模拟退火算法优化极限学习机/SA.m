function [X ,trace]=SA(P_train,P_test,T_train,T_test,initpop,gen,bound,D,hiddennum)
%% 参数设定  
%%%冷却表参数%%%%%%%%%%
L=2;      %马尔科夫链长度
K=0.8;    %衰减因子
S=0.01;     %步长因子
T=5;      %初始温度
temr=0.05;     %温度概率系数
P=0;        %Metroppolis过程中总接受点
Xs=bound(2);%上限
Xx=bound(1);%下限
trace=0;
YZ=1E-2;    %容差，相邻两次温度的差值
max_iter=gen;%最大退火次数                  %while循环用容差作为终止条件，for循环用最大退火次数作为终止条件
%% 产生初始最优解 与最优解对应的特征位置
Prex=initpop(:,1:end-1)';
fun=initpop(:,end)';

[sort_val,index_val] = sort(fun,'descend');
Prebestx=Prex(:,index_val(1));
Prex=Prex(:,index_val(2));
Bestx=Prebestx;

best_fitness=0;
%%
%每迭代一次退火一次(降温)，直到满足迭代条件为止
for iter=1:max_iter
    T=K*T;%在当前温度T下迭代次数
    for i=1:L
        Nextx=Prex;
        %在附近随机选下一点--核心思想：随机从特征中选10个，然后变成相对的数，比如0变1，1变0
        for j=1:10
            k=ceil(rand*length(Nextx));
            if Nextx(k)==0
                Nextx(k)=1;
            else 
                Nextx(k)=0;
            end
        end
        %%是否全局最优解
        a=fitness(Bestx,P_train,P_test,T_train,T_test,hiddennum);
        b=fitness(Nextx,P_train,P_test,T_train,T_test,hiddennum);
        if a<b
           Bestx=Nextx;   %更新最优解
           a=b;
        end%如果新解更好，用新解替代最优解，原最优解变为前最优解
        
%%%%%%%%%%%%Metropolis过程
%%% 这个过程是模拟退火的核心，以一定的概率接受次优解，意思就是用不好的解来替换最好解，来避免陷入局部最优解
%%% 这就会造成模拟退火算法的适应度曲线出现震荡，这也是和其他寻优算法的不同之处。
        c=fitness(Prex,P_train,P_test,T_train,T_test,hiddennum);
        if b>c %%%接受新解
            Prex=Nextx;
            P=P+1;
        else
            changer=-1/(temr*T);
            p1=exp(changer);
            if p1>rand%%%以一定概率接受较差的解
                Prex=Nextx;
                P=P+1;
            end
        end
    end
    %% 更新全局最优解-当当前最优解比记录的全局最优还要好，就更新至全局最优
     if a>best_fitness
           best_position=Bestx;
           best_fitness=a;
     end
     trace(iter+1)=best_fitness;
end
X=Bestx';