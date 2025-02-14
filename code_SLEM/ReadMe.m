clc;clear;close;
% Load the file containing the necessary inputs for calling the SLEM function
load('sample data.mat'); 

% trade-off parameters
gamma1=1;gamma2=1;lambda=1;%trade-off parameter

% Calling the main function SLEM
[ Eval,y_predict ] = SLEM(X_train,y_train,X_test,y_test,gamma1,gamma2,lambda);
disp(['HammingScore=',num2str(Eval.HS,'%4.3f'),', ExactMatch=',num2str(Eval.EM,'%4.3f'),', SubExactMatch=',num2str(Eval.SEM,'%4.3f')]);