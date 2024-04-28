
data_dir_new = 'D:\full_data-selected\preprocessed\after_spm_script\participant_';
data_dir_old = 'E:\PhD\participant_';
data_file = 'time_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';
addpath 'C:\External_Software\fieldtrip-20210807';

% load ft data
[data_old, ~] = load_data(data_dir_old, data_file);
[data_new, ~] = load_data(data_dir_new, data_file);

% calculate time window of interest. -500 to 300ms
start_time = -0.200;
end_time = 3;
electrode = 26; %A26
[new_avgTimeSeries, new_timeVector] = get_time_series_and_average(data_old, start_time, end_time, electrode, "Old");
[old_avgTimeSeries, old_timeVector] = get_time_series_and_average(data_new, start_time, end_time, electrode, "New");


plot(new_timeVector, new_avgTimeSeries);
xlabel('Time (ms)'); % Label the x-axis
ylabel('Average Time Series'); % Label the y-axis
title('Average Time Series vs. Time'); % Add a title
hold on;

plot(old_timeVector, old_avgTimeSeries);
xlabel('Time (ms)'); % Label the x-axis
ylabel('Average Time Series'); % Label the y-axis
title('Average Time Series vs. Time'); % Add a title
legend("Old TS", "New Ts");


disp('ok');


function [avgTimeSeries, timeVector]  = get_time_series_and_average(dataset, start_time, end_time, electrode, legend_name)
    time = dataset{1}.time;
    [~, start_idx] = min(abs(time - start_time));
    [~, end_idx] = min(abs(time - end_time));

    % Initialize a matrix to store all time series
    allTimeSeries = [];

    for i = 1:size(dataset, 2)
        data = dataset{i};
        timeseries = data.avg; 
        timeseries = timeseries(electrode, :);
        timeseries = timeseries(:, start_idx:end_idx);

        % Append the time series to the matrix
        allTimeSeries = [allTimeSeries; timeseries];
    end

    % Calculate the average time series
    avgTimeSeries = mean(allTimeSeries, 1);
    timeVector = time(start_idx:end_idx);
end


function [ft_regression_data,participant_order] = load_data(main_path, filename)
    n_participants = 40;
    partition.is_partition = 0;
    partition.partition_number = 0;

    ft_regression_data = {};
    participant_order = {};
    idx_used_for_saving_data = 1;
    for i=1:n_participants
        disp(strcat('LOADING PARTICIPANT...', int2str(i)));
        participant_main_path = strcat(main_path, int2str(i));
    
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