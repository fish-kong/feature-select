function Val = fitness(x,P_train,P_test,T_train,T_test,hiddennum)
[m,n] = find(x == 1);
p_train = [];
p_test = [];

for i = 1:length(n)
    p_train(:,i) = P_train(:,n(i));
    p_test(:,i) = P_test(:,n(i));
end %重新构建训练集、测试集、以及对应标签
t_train = T_train;
t_test=T_test;
[IW,B,LW,TF,TYPE,lambda,~] = elmtrain(p_train',t_train',hiddennum,'sig',1,inf);
predictlabel = elmpredict(p_test',IW,B,LW,TF,TYPE,lambda);
Val=sum(predictlabel==t_test')/length(predictlabel);
end
