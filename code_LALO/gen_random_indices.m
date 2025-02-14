%mirflickr:2780 lost:1122 fgnet:1002
num_samples = 1002;
indices = zeros(num_samples,10);
for col =1:10
    half_samples = num_samples / 2;
    indices1 = [ones(1, half_samples), 2 * ones(1, half_samples)];  
    indices1 = indices1(randperm(num_samples));  
    indices(:, col) = indices1;
end
save('random_indices_fgnet1.mat', 'indices');