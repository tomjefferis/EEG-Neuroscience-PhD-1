


clc; clear all;

path = '/Users/cihandogan/Documents/Research/preprocessing/after_spm_script/participant_';
n_participants = 40;
filename = 'time_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';
addpath('/Users/cihandogan/Documents/Research/fieldtrip-20230118');
partition.is_partition = 0;
partition.partition_number = 0;
participants = [16];

[dataset, participant_order] = load_postprocessed_data(path, n_participants, filename, partition, participants);

%path = '/Users/cihandogan/Documents/Research/PhD/participant_';
%filename = 'time_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';

%[dataset_two, participant_order] = load_postprocessed_data(path, n_participants, filename, partition, participants);

%participants = [13];
%path = '/Users/cihandogan/Documents/Research/preprocessing/after_spm_script/participant_';
%[dataset_three, participant_order] = load_postprocessed_data(path, n_participants, filename, partition, participants);

cfg = [];
x = dataset{1};
ft_multiplotER(cfg, x);

%old_participant = dataset_two{1}.avg(23, :);
%new_participant = dataset{1}.avg(23, :);
%new_participant_13 = dataset_three{1}.avg(23, :);
%%time_old = dataset_two{1}.time;
%time_new = dataset{1}.time;
%time_new_13 = dataset_three{1}.time;

% Create a new time vector that starts at -0.5 and ends at 0.3
%time = linspace(-0.5, 0.3, 500)';

%old_participant_interp = interp1(time_old, old_participant, time);
%new_participant_interp = interp1(time_new, new_participant, time);
%new_participant_13_interp = interp1(time_new_13, new_participant_13, time);


%figure;
%plot(time, old_participant_interp, 'DisplayName', 'Old Participant 23'); hold on;
%plot(time, new_participant_interp, 'DisplayName', 'New Participant 23');
%plot(time, new_participant_13_interp, 'DisplayName', 'New Participant 13')
%legend('Location', 'northwest');
%hold off;

%disp('hello');

%% load post-processed fildtrip data
function [ft_regression_data, participant_order] = ...
    load_postprocessed_data(main_path, n_participants, filename, partition, participants)

ft_regression_data = {};
participant_order = {};

idx_used_for_saving_data = 1;
for i=participants
    disp(strcat('LOADING PARTICIPANT...', int2str(i)));
    participant_main_path = strcat(main_path, int2str(i));

    %exp_type = 'mean_intercept';

    %if strcmp(exp_type, 'mean_intercept')
    %   parts_to_filter = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,38,39,40];
    %else
    %   parts_to_filter = [1,2,3,4,6,7,8,9,10,12,13,14,17,20,21,22,23,25,26,30,31,32,33,34,38,39,40];
    %end
    
    %if ismember(i,parts_to_filter)



    if exist(participant_main_path, 'dir')
        cd(participant_main_path);

        if isfile(filename)
            load(filename);
        else
            continue;
        end

        ft.label = data.label;
        ft.time = data.time{1};
        ft.trialinfo = [1];
        ft.elec = data.elec;
        ft.dimord = 'chan_time';

        % find the condition labels used to match up the data

        if partition.is_partition
            if partition.partition_number == 1
                if isfield(data, 'p1_pgi')
                    pgi = data.p1_pgi;
                end
                thin = data.p1_thin;
                med = data.p1_med;
                thick = data.p1_thick;
            elseif partition.partition_number == 2
                if isfield(data, 'p2_pgi')
                    pgi = data.p2_pgi;
                end
                thin = data.p2_thin;
                med = data.p2_med;
                thick = data.p2_thick;
            elseif partition.partition_number == 3
                if isfield(data, 'p3_pgi')
                    pgi = data.p3_pgi;
                end
                thin = data.p3_thin;
                med = data.p3_med;
                thick = data.p3_thick;
            else
                thin = data.thin;
                med = data.med;
                thick = data.thick;
            end
        elseif ~partition.is_partition
            try
                pgi = data.med - (data.thin + data.thick)/2;
                ft.avg = pgi;
            catch
                disp('freq')
            end
            thin = data.thin;
            med = data.med;
            thick = data.thick;

        end

        if isfield(data, 'p1_pgi') || isfield(data, 'p2_pgi') || isfield(data, 'p3_pgi')
            ft.avg = pgi;
        end

        ft.thin = thin;
        ft.med = med;
        ft.thick = thick;

        if isfield(data, 'thick_order') || isfield(data, 'thin_order') ...
                || isfield(data, 'med_order')
            ft.thick_order = data.thick_order;
            ft.thin_order = data.thin_order;
            ft.med_order = data.med_order;
        end

        ft = rmfield(ft, "trialinfo");

        ft_regression_data{idx_used_for_saving_data} = ft;
        participant_order{idx_used_for_saving_data} = i;
        idx_used_for_saving_data = idx_used_for_saving_data + 1;
    end
    end
end