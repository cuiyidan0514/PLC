num_samples = 17472;
indices = zeros(num_samples,10);
for col =1:10
    half_samples = num_samples / 2;
    indices1 = [ones(1, half_samples), 2 * ones(1, half_samples)];  
    indices1 = indices1(randperm(num_samples));  
    indices(:, col) = indices1;
end
save('random_indices_soccer.mat', 'indices');