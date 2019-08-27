%% 1����ջ���
clear;close all; format compact
%% 2��ģ���˻���Ż����㼫��ѧϰ����������ѡ�� ɾ����������

%% ��������
load data
P_train=wdbc(1:500,1:30);
T_train=wdbc(1:500,31);
P_test=wdbc(501:end,1:30);
T_test=wdbc(501:end,31);

%% elm����
hiddennum=50;

%%
rand('seed',0)
[IW,B,LW,TF,TYPE,lambda,train_accuracy] = elmtrain(P_train',T_train',hiddennum,'sig',1,inf);
predictlabel = elmpredict(P_test',IW,B,LW,TF,TYPE,lambda);
acc1=sum(predictlabel==T_test')/length(predictlabel);
figure
stem(predictlabel);hold on
plot(T_test,'*')
title('δ����ѡ��֮ǰ��ELM���Լ�����')
xlabel('�������')
ylabel('����ǩ')

%%
%% SAģ���˻��㷨�Ż� ������ 
D=size(P_train,2);          %���ݵ�ά�ȡ���������
popu = 5;                   %ģ���˻��㷨��ʼ�����
bound =[0 1];              %�߽�����,Ҫô��0��Ҫô��1
gen = 100;                   %��������

% ������ʼ��Ⱥ
initPop = randi([0 1],popu,D);
% �����ʼ��Ⱥ��Ӧ��
initFit = zeros(popu,1);
for i = 1:size(initPop,1)
    initFit(i) = fitness(initPop(i,:),P_train,P_test,T_train,T_test,hiddennum);
end
initPop = [initPop initFit];

%%
[X,trace]=SA(P_train,P_test,T_train,T_test,initPop,gen,bound,D,hiddennum);
[m,n]=find(X==1);
disp(['�Ż�ɸѡ��������Ա������Ϊ:' num2str(n)]);
%%
figure
plot(trace,'r*-');
title('��Ӧ�Ⱥ���')
xlabel('������')
ylabel('��Ӧ��(���Լ�����׼ȷ��)')
%% �����Ż�����������ѵ��������
%%��ѵ����/���Լ�������ȡ
p_train = [];
p_test = [];
for i = 1:length(n)
    p_train(:,i) = P_train(:,n(i));
    p_test(:,i) = P_test(:,n(i));
end
t_train = T_train;
t_test=T_test;

[IW,B,LW,TF,TYPE,lambda,train_accuracy] = elmtrain(p_train',t_train',hiddennum,'sig',1,inf);
predictlabe2 = elmpredict(p_test',IW,B,LW,TF,TYPE,lambda);
acc2=sum(predictlabe2==t_test')/length(predictlabe2);
figure
stem(predictlabe2);hold on
plot(t_test,'*')
title('����ѡ��֮���ELM���Լ�����')
xlabel('�������')
ylabel('����ǩ')

disp('չʾû�н���ģ���˻��Ż��Ա�����άǰ�ļ���ѧϰ��������ȷ��')
acc1
disp('չʾģ���˻��Ż��Ա�����ά��ļ���ѧϰ��������ȷ��')
acc2
