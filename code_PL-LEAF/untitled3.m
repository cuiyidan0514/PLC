acc = [6.6,65.9,46.5,62.2,51.9,60.8];
acc_best = [7.5,70.2,49.0,62.4,52.6,62.4];
acc_best1 = [0.641670,0.644367,0.643758,0.646368,0.636016,0.643062,0.639669,0.647182,0.639179,0.635668];

acc_mean = mean(acc);
acc_std = std(acc);
acc_best_mean = mean(acc_best);
acc_best_std = std(acc_best);
acc_best_mean1 = mean(acc_best1);
acc_best_std1 = std(acc_best1);

disp(acc);
disp(acc_best);

fprintf('Accuracy mean: %.3f, std: %.3f\n', acc_mean, acc_std);  
fprintf('Accuracy (SVD best) mean: %.3f, std: %.3f\n', acc_best_mean, acc_best_std);  
fprintf('Accuracy (SVD best,1<k<5) mean: %.3f, std: %.3f\n', acc_best_mean1, acc_best_std1); 