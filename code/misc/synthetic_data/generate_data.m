clear all;clc

%% settings for synth data
samples = 2001;
num_participants = 40;
n_trials = 500;
total_trials = num_participants*n_trials;
width_lag = 160;
toi = [-0.2, 0.8];


%% generate X amount of signals
t = [0:1/(samples-1):1]; 
x = sin(2*pi*t) + sin(4*pi*t) + sin(8*pi*t); 
y = exp(0.01*[-1*[(samples/2):-1:1] 0 -1*[1:(samples/2)]]); 
reference = x.*y; 
synth_data = generate_signals(reference, samples, total_trials, width_lag);

%% generate some pink noise
sumsig = 3;
pink_noise = noise(samples,total_trials,samples,3);

%% add the pink noise on top of the sythetic data
signals = zeros(samples, total_trials);
for t = 1:total_trials
    pink_noise_j = pink_noise(:,t);
    synth_data_j = synth_data(:,t);
    sig_w_pink_noise = synth_data_j + pink_noise_j;
    signals(:,t) = sig_w_pink_noise;
end

%% create synth participants and generate their ERPs
participants = {};
k_trials = n_trials;
for p = 1:num_participants
    
    if p == 1
        subset = signals(:,1:k_trials);
    else
        subset = signals(:,k_trials+1:k_trials + (n_trials));
        k_trials = k_trials + n_trials;
    end
    
    erp = mean(subset,2);
    data.erp = erp;
    data.trials = subset;
    participants{p} = data;
end

%% create spectrograms using morlett waveletts on both the trial and ERP leve
cfg              = [];
cfg.output       = 'pow';
cfg.method       = 'wavelet';
cfg.taper        = 'hanning';
cfg.width = 3;
cfg.foi =   5:30;
cfg.t_ftimwin = ones(length(cfg.foi),1).*0.25;
cfg.toi          = toi(1):0.002:toi(2);

end_value = toi(2);  
start_value = toi(1);
n_elements = samples;
step_size = (end_value-start_value)/(n_elements-1);

for p=1:num_participants
    data = participants{p};
    erp = data.erp;
    trials = data.trials;
    time = start_value:step_size:end_value;
    
    % erp level
    erp_level.dimord = 'chan_time';
    erp_level.trial = erp';
    erp_level.elec = {};
    erp_level.label = {'A1'};
    erp_level.time = time;
    erp_tf = ft_freqanalysis(cfg, erp_level);
    
    % trial-level
    trial_level.dimord = 'chan_time';
    trial_level.trial = create_ft_data(n_trials, trials);
    trial_level.elec = {};
    trial_level.label = {'A1'};
    trial_level.time = create_fieldtrip_format(n_trials,time);
    tl_tf = ft_freqanalysis(cfg, trial_level);
end


cfg = [];
cfg.baseline = 'no';
cfg.xlim = [toi(1),toi(2)];
cfg.channel = 'A1';
ft_singleplotTFR(cfg, tl_tf);


%% converts to a FT format
function data = create_fieldtrip_format(n, series)
    data = {};
    for k = 1:n
        data{k} = series;
    end
end

function dataset = create_ft_data(n, data)
    dataset = {};
    data = data';
    for k =1:n
        dataset{k} = data(k,:);
    end
end