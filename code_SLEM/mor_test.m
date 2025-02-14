function [degree,degree_SVD] = mor_test(testX, trainX, train_p_target, model, verbose)
%mor_test implements the testing procedure of multi-output regression
    if nargin<4
        verbose = 0;
    end

    if verbose
        fprintf(1,'\nmor_test begins...\n');
    end

    %Compute kernel matrix for prediction using testX and trainX
    Ktest = KerMtx(model.par, testX, trainX);
    ktrain = KerMtx(model.par, trainX, trainX);
    %Prediction.
    degree = Ktest*model.Beta+repmat(model.b,size(Ktest,1),1);
    
    %SVD
    degree_SVD = optimize_output(trainX, train_p_target, testX, model, ktrain, Ktest, 3);
end