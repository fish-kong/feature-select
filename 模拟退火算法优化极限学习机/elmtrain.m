function [IW,B,LW,TF,TYPE,lambda,train_accuracy] = elmtrain(P_train,T,N,TF,TYPE,lambda)

%P_train=train;T=train_label;N=200;TF='sig';TYPE=1;
% P   - Input Matrix of Training Set  (R*Q)
% T   - Output Matrix of Training Set (S*Q)
% N   - Number of Hidden Neurons (default = Q)
% TF  - Transfer Function:'sig' for Sigmoidal function (default)',sin' for Sine function,'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)

% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (N*S)
%lambda-L1 regularation
%�ع�
% [IW,B,LW,TF,lambda] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% ����
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% See also ELMPREDICT


%��Ȼ�����������elmtrain����6�����������������nargin=6��������Щʡ�Ե�ʱ�򣬾�Ҫ�õ����漸��������Ĭ�ϸ�ֵ
if nargin < 2
    error('ELM:Arguments','Not enough input arguments.');
end  %�������������ڵ���2���������޷����н�ģ����Ϊ����Ҫ����������������
if nargin < 3
    N = size(P_train,2); %���ֻ�����������Ĭ����������ԪΪ������
end
if nargin < 4
    TF = 'sig';%���ֻ��������룬��������Ԫ������Ĭ�ϼ����Ϊsigmoid����
end
if nargin < 5
    TYPE = 0;%���û�ж��庯�������ã�Ĭ��Ϊ�ع����
end
% ���л����϶������������ж������Ǽ��������ģ����������⼸��һ���ò���������������������
%%%%%%%%%%%%*****************************

if size(P_train,2) ~= size(T,2)
    error('ELM:Arguments','The columns of P and T must be same.');
end
%����������������������������һ�¡�
[R,Q] = size(P_train);%R=2,Q=1900
T1=T;
if TYPE  == 1
    T1  = ind2vec(T);
end
                         %���������Ƿ��࣬�ͽ�ѵ�����תΪ��������   http://blog.csdn.net/u011314012/article/details/51191006
[S,Q] = size(T1); %S=1,Q=1900

% �����������Ȩ�ؾ���1900*2
IW = rand(N,R) * 2 - 1;

% �����������ƫ�� 1900*1
B = rand(N,1);
BiasMatrix = repmat(B,1,Q);

% �����������H
tempH = IW * P_train + BiasMatrix;
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end
% �������㵽�����֮���Ȩ��
LW = pinv((H+1/lambda)') * T1';
TY=(H'*LW)';
Y_train=TY;
if TYPE  == 1
    temp_Y=zeros(1,size(TY,2));
    for n=1:size(TY,2)
        [max_Y,index]=max(TY(:,n));
        temp_Y(n)=index;
    end
    Y_train=temp_Y;
    train_accuracy=sum(Y_train==T)/length(T);
end
if TYPE==0
    train_accuracy=0;
end
%figure(1)
%plot(Y_train,'r*');hold on
%plot(T,'bo');
%title('ѵ����')
end

%pinv��inv������������������󣬵���inv��֪������������������õ�
%������������Ƿ�������������ʱ�����߸�������������󣬾���PINV��α�����