clc
close all
clear all

load('reward.mat');
r = reward;
t = 3000;
a = 100
for i = 1:t
    k1 = (i-1)*a+1;
    k2 = i*a;
    y = r(k1:k2,1);
    x = sum(y)/a;
    H(1,i) = x;
    

end

%%
%第一张图
pic = figure(1);
grid on;
hold on;
% ax = gca;
% ax.YScale = 'log';
xlabel('epoches');
ylabel('Average Reward','FontSize',18);
pic.NumberTitle = 'off';


X = 1:1:t;




plot(X,H(1,:),'b-');

legend('Reward','Adaptive MCS and payload length, STA','Location','Northwest');

axis([0 t 0 1])
hold off;