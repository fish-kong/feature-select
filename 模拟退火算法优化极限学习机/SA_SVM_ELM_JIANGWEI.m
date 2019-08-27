%% 1、清空环境
clear;close all; format compact
%% 2、模拟退火的优化计算极限学习机进行特征选择 删除冗余特征

%% 加载数据
load data
P_train=wdbc(1:500,1:30);
T_train=wdbc(1:500,31);
P_test=wdbc(501:end,1:30);
T_test=wdbc(501:end,31);

%% elm参数
hiddennum=50;

%%
rand('seed',0)
[IW,B,LW,TF,TYPE,lambda,train_accuracy] = elmtrain(P_train',T_train',hiddennum,'sig',1,inf);
predictlabel = elmpredict(P_test',IW,B,LW,TF,TYPE,lambda);
acc1=sum(predictlabel==T_test')/length(predictlabel);
figure
stem(predictlabel);hold on
plot(T_test,'*')
title('未特征选择之前的ELM测试集精度')
xlabel('样本标号')
ylabel('类别标签')

%%
%% SA模拟退火算法优化 主程序 
D=size(P_train,2);          %数据的维度——特征数
popu = 5;                   %模拟退火算法初始解个数
bound =[0 1];              %边界条件,要么是0，要么是1
gen = 100;                   %迭代次数

% 产生初始种群
initPop = randi([0 1],popu,D);
% 计算初始种群适应度
initFit = zeros(popu,1);
for i = 1:size(initPop,1)
    initFit(i) = fitness(initPop(i,:),P_train,P_test,T_train,T_test,hiddennum);
end
initPop = [initPop initFit];

%%
[X,trace]=SA(P_train,P_test,T_train,T_test,initPop,gen,bound,D,hiddennum);
[m,n]=find(X==1);
disp(['优化筛选后的输入自变量编号为:' num2str(n)]);
%%
figure
plot(trace,'r*-');
title('适应度函数')
xlabel('迭代数')
ylabel('适应度(测试集分类准确率)')
%% 利用优化的特征重新训练并测试
%%新训练集/测试集数据提取
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
title('特征选择之后的ELM测试集精度')
xlabel('样本标号')
ylabel('类别标签')

disp('展示没有进行模拟退火优化自变量降维前的极限学习机分类正确率')
acc1
disp('展示模拟退火优化自变量降维后的极限学习机分类正确率')
acc2
