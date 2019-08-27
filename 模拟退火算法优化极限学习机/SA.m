function [X ,trace]=SA(P_train,P_test,T_train,T_test,initpop,gen,bound,D,hiddennum)
%% �����趨  
%%%��ȴ�����%%%%%%%%%%
L=2;      %����Ʒ�������
K=0.8;    %˥������
S=0.01;     %��������
T=5;      %��ʼ�¶�
temr=0.05;     %�¶ȸ���ϵ��
P=0;        %Metroppolis�������ܽ��ܵ�
Xs=bound(2);%����
Xx=bound(1);%����
trace=0;
YZ=1E-2;    %�ݲ���������¶ȵĲ�ֵ
max_iter=gen;%����˻����                  %whileѭ�����ݲ���Ϊ��ֹ������forѭ��������˻������Ϊ��ֹ����
%% ������ʼ���Ž� �����Ž��Ӧ������λ��
Prex=initpop(:,1:end-1)';
fun=initpop(:,end)';

[sort_val,index_val] = sort(fun,'descend');
Prebestx=Prex(:,index_val(1));
Prex=Prex(:,index_val(2));
Bestx=Prebestx;

best_fitness=0;
%%
%ÿ����һ���˻�һ��(����)��ֱ�������������Ϊֹ
for iter=1:max_iter
    T=K*T;%�ڵ�ǰ�¶�T�µ�������
    for i=1:L
        Nextx=Prex;
        %�ڸ������ѡ��һ��--����˼�룺�����������ѡ10����Ȼ������Ե���������0��1��1��0
        for j=1:10
            k=ceil(rand*length(Nextx));
            if Nextx(k)==0
                Nextx(k)=1;
            else 
                Nextx(k)=0;
            end
        end
        %%�Ƿ�ȫ�����Ž�
        a=fitness(Bestx,P_train,P_test,T_train,T_test,hiddennum);
        b=fitness(Nextx,P_train,P_test,T_train,T_test,hiddennum);
        if a<b
           Bestx=Nextx;   %�������Ž�
           a=b;
        end%����½���ã����½�������Ž⣬ԭ���Ž��Ϊǰ���Ž�
        
%%%%%%%%%%%%Metropolis����
%%% ���������ģ���˻�ĺ��ģ���һ���ĸ��ʽ��ܴ��Ž⣬��˼�����ò��õĽ����滻��ý⣬����������ֲ����Ž�
%%% ��ͻ����ģ���˻��㷨����Ӧ�����߳����𵴣���Ҳ�Ǻ�����Ѱ���㷨�Ĳ�֮ͬ����
        c=fitness(Prex,P_train,P_test,T_train,T_test,hiddennum);
        if b>c %%%�����½�
            Prex=Nextx;
            P=P+1;
        else
            changer=-1/(temr*T);
            p1=exp(changer);
            if p1>rand%%%��һ�����ʽ��ܽϲ�Ľ�
                Prex=Nextx;
                P=P+1;
            end
        end
    end
    %% ����ȫ�����Ž�-����ǰ���Ž�ȼ�¼��ȫ�����Ż�Ҫ�ã��͸�����ȫ������
     if a>best_fitness
           best_position=Bestx;
           best_fitness=a;
     end
     trace(iter+1)=best_fitness;
end
X=Bestx';