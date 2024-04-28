% print scientific notation
format longG

% load factor data
factors = readtable('factors.csv');

% get vs factors
vsq = factors.VSQ_z_;
chi = factors.TotalChi_z_;
aura = factors.Aura_z_;

% combine all three variables
visual_stress = [vsq, chi, aura];

% apply a PCA on all data - same exact 1:1 components in Python
% take the first component and see how that vector of three changes.
[coeff, ~, ~, ~, ~, mu] = pca(visual_stress);

% Subtract the mean of the training data from the input data
%visual_stress_centered = bsxfun(@minus, visual_stress, mu);

% Transform the input data using the PCA model
visual_stress_pca = visual_stress_centered * coeff;

% take the mean of the columns
means = mean(visual_stress_pca);

%% seperately perform bootstrapping
num_bootstraps = 50000;


bootstrapped_means = zeros(num_bootstraps, size(visual_stress, 2));
bootstrapped_means_avg = zeros(num_bootstraps, size(visual_stress, 2));
for i = 1:num_bootstraps
    disp(i)
    % i get my bootstrap sample
    bootstrap_sample = datasample(visual_stress, size(visual_stress, 1));
    
    bootstrapped_means_avg(i, :) = mean(bootstrap_sample);


    % apply a PCA on all data - same exact 1:1 components in Python
    [coeff, ~, ~, ~, ~, mu] = pca(bootstrap_sample);

    % Subtract the mean of the training data from the input data
    %bootstrap_sample_centered = bsxfun(@minus, bootstrap_sample, mu);
    
    % Transform the input data using the PCA model
    bootstrap_sample_pca = bootstrap_sample * coeff;
    
    % take the mean of the columns
    means = mean(bootstrap_sample_pca);

    % append to our matrix
    bootstrapped_means(i, :) = means;
end

% print variances
disp('pca')
disp(var(bootstrapped_means));
disp(std(bootstrapped_means));

disp('averages')
disp(var(bootstrapped_means_avg));
disp(std(bootstrapped_means_avg));





