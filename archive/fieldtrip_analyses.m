%% suplementary experiment inforamtion
% ft topoplot
clear classes;
master_dir = 'C:\ProgramFiles\PhD\fieldtrip';
main_path = 'C:\ProgramFiles\PhD\participant_';
results_dir = 'C:\ProgramFiles\PhD\results';
rmpath C:\ProgramFiles\spm8;
addpath C:\ProgramFiles\spm12;
cd(master_dir);
experiment_types = {'partitions-2-8'};   
desired_design_mtxs = {'no-factor', 'headache', 'visual_stress', 'discomfort'};

start_latency = 0.056;
end_latency = 0.256;

region_of_interest = 1;
roi_applied = 'two-tailed';
weight_roi = 0;
roi_to_apply = 0;

generate_erps = 1;
bootstrap_ci_erps = 1;

for i = 1:numel(experiment_types)
    for j = 1:numel(desired_design_mtxs)
        experiment_type = experiment_types{i};
        desired_design_mtx = desired_design_mtxs{j};
        %% create the results save path depending on the experiment
        if contains(experiment_type, 'partitions') || contains(experiment_type, 'Partitions')
            save_path = strcat(results_dir, '\', 'partitions', '\', desired_design_mtx);
        elseif contains(experiment_type, 'erps-23-45-67')
            save_path = strcat(results_dir, '\', 'onsets', '\', desired_design_mtx);
        elseif contains(experiment_type, 'onsets-2-8-explicit')
            save_path = strcat(results_dir, '\', 'mean_intercept', '\', desired_design_mtx);
        end
        
        %% Are we looking at onsets 2-8 or partitions
        % set up the experiment as needed
        if strcmp(experiment_type, 'onsets-2-8-explicit')
            data_file = 'grand-avg_trial-level_mean_interceptb1f1';
            regressor = 'ft_statfun_depsamplesT';
            n_participants = 40;
            start_latency = 0.056;
            end_latency = 0.256;

            partition.is_partition = 0;
            partition.partition_number = 0;

            [data, participant_order] = from_fieldtrip_to_spm(n_participants,main_path,data_file,partition, 'regression');
            n_part = numel(data);
            design_matrix =  [1:n_part 1:n_part; ones(1,n_part) 2*ones(1,n_part)]; 

        elseif strcmp(experiment_type, 'onsets-1-explicit')
            data_file = 'averaged_onsets_1t1b1f1';
            regressor = 'ft_statfun_depsamplesT';
            n_participants = 40;

            partition.is_partition = 0;
            partition.partition_number = 0;

            [data, participant_order] = from_fieldtrip_to_spm(n_participants,main_path,data_file,partition, 'regression');
            n_part = numel(data);
            design_matrix =  [1:n_part 1:n_part; ones(1,n_part) 2*ones(1,n_part)];

        elseif strcmp(experiment_type, 'onsets-2-8-factor') || strcmp(experiment_type, 'onsets-1-factor')
            if strcmp(experiment_type, 'onsets-2-8-factor')
                data_file = 'averaged_onsets_2_3_4_5_6_7_8t1b1f1';
            elseif strcmp(experiment_type, 'onsets-1-factor')
                data_file = 'averaged_onsets_1t1b1f1';
            end

            regressor = 'ft_statfun_indepsamplesregrT';
            n_participants = 40;   
            regression_type = desired_design_mtx;
            start_latency = 0.056;
            end_latency = 0.256;

            partition.is_partition = 0;
            partition.partition_number = 0;
            [data, participant_order] = from_fieldtrip_to_spm(n_participants,main_path,data_file,partition, 'regression');
            n_part = numel(data);

            [design_matrix, data] = create_design_matrix_partitions(participant_order, data, ...
                regression_type, 1);

            if region_of_interest == 1
                if strcmp(experiment_type, 'onsets-2-8-factor')
                    if strcmp(roi_applied, 'one-tailed')
                        load('C:\ProgramFiles\PhD\fieldtrip\one_tailed_roi_28.mat');
                    elseif strcmp(roi_applied, 'two-tailed')
                        load('C:\ProgramFiles\PhD\fieldtrip\two_tailed_roi_28.mat');
                    end
                    %start_latency = 0.083;
                    %end_latency = 0.256;
                elseif strcmp(experiment_type, 'onsets-1-factor')
                    if strcmp(roi_applied, 'one-tailed')
                        load('C:\ProgramFiles\PhD\fieldtrip\roi\one_tailed_roi_1.mat');
                    elseif strcmp(roi_applied, 'two-tailed')
                        load('C:\ProgramFiles\PhD\fieldtrip\roi\two_tailed_roi_1.mat');
                    end

                    %start_latency = 0.0163;
                    %end_latency = 0.256;
                end
                data = create_hacked_roi(data, roi, weight_roi);
            end

        elseif strcmp(experiment_type, 'partitions-2-8') || strcmp(experiment_type, 'partitions-1') 
            if strcmp(experiment_type, 'partitions-2-8') 
                data_file = 'grand-avg_trial-level_partitionsb1f1';
            elseif strcmp(experiment_type, 'partitions-1')
                data_file = 'grand-avg_trial-level_partitionsb1f1';
            end
            type_of_effect = 'habituation';
            regressor = 'ft_statfun_indepsamplesregrT';
            regression_type = desired_design_mtx;
            n_participants = 39;

            partition1.is_partition = 1; % partition 1
            partition1.partition_number = '1';
            partition2.is_partition = 1; % partition 2
            partition2.partition_number = '2';
            partition3.is_partition = 1; % partition 3
            partition3.partition_number = '3';

            [data1, participant_order_1] = from_fieldtrip_to_spm(n_participants,main_path,data_file,partition1, 'regression');
            cd(master_dir);
            [data2, participant_order_2] = from_fieldtrip_to_spm(n_participants,main_path,data_file,partition2, 'regression');
            cd(master_dir);
            [data3, participant_order_3] = from_fieldtrip_to_spm(n_participants,main_path,data_file,partition3, 'regression');
            cd(master_dir);

            partition = 1;
            [design1, new_participants1] = create_design_matrix_partitions(participant_order_1, data1, ...
                    regression_type, partition, type_of_effect);
            partition = 2;
            [design2, new_participants2] = create_design_matrix_partitions(participant_order_2, data2, ...
                    regression_type, partition, type_of_effect);
            partition = 3;
            [design3, new_participants3] = create_design_matrix_partitions(participant_order_3, data3, ...
                    regression_type, partition, type_of_effect);
            data = [new_participants1, new_participants2, new_participants3];
            design_matrix = [design1, design2, design3];
            n_part = numel(data);
            n_part_per_desgin = numel(design1);

            if strcmp(desired_design_mtx, 'no-factor')
                design1(1:numel(design1)) = 2.72;
                design2(1:numel(design1)) = 1.65;
                design3(1:numel(design1)) = 1.00;
                design_matrix = [design1, design2, design3];
                min_x = min(design_matrix);
                design_matrix = design_matrix - min_x;
            end

            % to fix 
            if strcmp(desired_design_mtx, 'headache-scores-entire-median-split') ...
                    || strcmp(desired_design_mtx, 'aura-scores-entire-median-split')
                newdesign1 = design1 * 2.72;
                newdesign2 = design2 * 1.65;
                newdesign3 = design3 * 1.00;
                design_matrix = [newdesign1, newdesign2, newdesign3];
                min_x = min(design_matrix);
                design_matrix = design_matrix - min_x;
            end
            
            design_matrix = design_matrix - mean(design_matrix);
            save_desgin_matrix(design_matrix, n_part_per_desgin, save_path, 'habituation')

            if region_of_interest == 1
                if strcmp(experiment_type, 'partitions-2-8') 
                    if strcmp(roi_applied, 'one-tailed')
                        load('C:\ProgramFiles\PhD\fieldtrip\roi\one_tailed_roi_28.mat');
                    elseif strcmp(roi_applied, 'two-tailed')
                        load('C:\ProgramFiles\PhD\fieldtrip\roi\two_tailed_roi_28.mat');
                    end
                end
                data = create_hacked_roi(data, roi, weight_roi);
            end

        elseif strcmp(experiment_type, 'erps-23-45-67') 
            data_file23 = 'averaged_onsets_2_3t1b1f1';
            data_file45 = 'averaged_onsets_4_5t1b1f1';
            data_file67 = 'averaged_onsets_6_7t1b1f1';  

            n_participants = 40;
            type_of_effect = 'sensitization';
            data_loader.is_partition = 0;
            data_loader.partition_number = 0;
            regressor = 'ft_statfun_indepsamplesregrT';
            regression_type = desired_design_mtx;

            [data1, participant_order_1] = from_fieldtrip_to_spm(n_participants,main_path,data_file23,data_loader, 'regression');
            [data2, participant_order_2] = from_fieldtrip_to_spm(n_participants,main_path,data_file45,data_loader, 'regression');
            [data3, participant_order_3] = from_fieldtrip_to_spm(n_participants,main_path,data_file67,data_loader, 'regression');

            
            partition = 1;
            [design1, new_participants1] = create_design_matrix_partitions(participant_order_1, data1, ...
                    regression_type, partition, type_of_effect);
            partition = 2;
            [design2, new_participants2] = create_design_matrix_partitions(participant_order_2, data2, ...
                    regression_type, partition, type_of_effect);
            partition = 3;
            [design3, new_participants3] = create_design_matrix_partitions(participant_order_3, data3, ...
                    regression_type, partition, type_of_effect);

            data = [new_participants1, new_participants2, new_participants3];
            design_matrix = [design1, design2, design3];
            n_part = numel(data);
            n_part_per_desgin = numel(design1);


            if strcmp(desired_design_mtx, 'no-factor')
                design_matrix_a(1:numel(design1)) = 1.00;
                design_matrix_b(1:numel(design1)) = 1.65;
                design_matrix_c(1:numel(design1)) = 2.72;
                design_matrix = [design_matrix_a, design_matrix_b, design_matrix_c];
                min_x = min(design_matrix);
                design_matrix = design_matrix - min_x;
            end

            if strcmp(desired_design_mtx, 'headache-scores-entire-median-split') ... 
                    || strcmp(desired_design_mtx, 'discomfort-scores-entire-median-split')
                newdesign1 = design1 * 1.00;
                newdesign2 = design2 * 1.65;
                newdesign3 = design3 * 2.72;
                design_matrix = [newdesign1, newdesign2, newdesign3];
                min_x = min(design_matrix);
                design_matrix = design_matrix - min_x;  
            end

            design_matrix = design_matrix - mean(design_matrix);
            save_desgin_matrix(design_matrix, n_part_per_desgin, save_path, 'sensitization')

            if region_of_interest == 1
                if strcmp(experiment_type, 'erps-23-45-67') || strcmp(experiment_type,  'erps-23-45-67-no-factor')
                    if strcmp(roi_applied, 'one-tailed')
                        load('C:\ProgramFiles\PhD\fieldtrip\roi\one_tailed_roi_28.mat');
                    elseif strcmp(roi_applied, 'two-tailed')
                        load('C:\ProgramFiles\PhD\fieldtrip\roi\two_tailed_roi_28.mat');
                    end
                end
                data = create_hacked_roi(data, roi, weight_roi);
            end
        end

        %% setup FT analysis
        % we have to switch to SPM8 to use some of the functions in FT
        rmpath C:\ProgramFiles\spm12;
        addpath C:\ProgramFiles\spm8;

        % we need to tell fieldtrip how our electrodes are structured
        cfg = [];
        cfg.feedback = 'no';
        cfg.method = 'distance';
        cfg.elec = data{1}.elec;
        neighbours = ft_prepare_neighbours(cfg);

        % all experiment configuraitons:
        cfg = [];
        cfg.latency = [start_latency, end_latency];
        cfg.channel = 'eeg';
        cfg.statistic = regressor;
        cfg.method = 'montecarlo';
        cfg.correctm = 'cluster';
        cfg.neighbours = neighbours;
        cfg.clusteralpha = 0.025;
        cfg.numrandomization = 25000;
        cfg.tail = roi_to_apply; 
        cfg.design = design_matrix;
        %cfg.computeprob = 'yes';
        %cfg.correcttail = 'alpha'; 

        %% run the fieldtrip analyses
        if contains(experiment_type, 'onsets-2-8-explicit') || contains(experiment_type, 'onsets-1-explicit')
            cfg.uvar = 1;
            cfg.ivar = 2;
            null_data = set_values_to_zero(data); % create null data to hack a t-test
            stat = ft_timelockstatistics(cfg, data{:}, null_data{:});
            desired_cluster =1;
            get_region_of_interest_electrodes(stat, desired_cluster, experiment_type, roi_applied);
        elseif contains(experiment_type, 'partitions') || contains(experiment_type, 'onsets-2-8-factor') ...
                || contains(experiment_type, 'onsets-1-factor') || contains(experiment_type, 'erps-23-45-67') ...
                || contains(experiment_type, 'coarse-vs-fine-granularity') || contains(experiment_type, 'Partitions')
            cfg.ivar = 1;
            stat = ft_timelockstatistics(cfg, data{:});
        end

        %% get peak level stats
        % 1 for the most positive going cluster
        if cfg.tail == 1
            [pos_peak_level_stats, pos_all_stats] = get_peak_level_stats(stat, 1, 'positive');
        elseif cfg.tail == 0
            [pos_peak_level_stats, pos_all_stats] = get_peak_level_stats(stat, 1, 'positive');
            [neg_peak_level_stats, neg_all_stats] = get_peak_level_stats(stat, 1, 'negative');
        end

        %% generate ERPs using the stat information
        if generate_erps == 1
            if cfg.tail == 1
                generate_peak_erps(master_dir, main_path, experiment_type, ...
                    stat, pos_peak_level_stats, 'positive', desired_design_mtx, 1, ...
                    save_path, bootstrap_ci_erps);
            elseif cfg.tail == 0
                generate_peak_erps(master_dir, main_path, experiment_type, ...
                    stat, pos_peak_level_stats, 'positive', desired_design_mtx, 1, ...
                    save_path, bootstrap_ci_erps);
                generate_peak_erps(master_dir, main_path, experiment_type, ...
                    stat, neg_peak_level_stats, 'negative', desired_design_mtx, 1, ...
                    save_path, bootstrap_ci_erps);
            end
        end
        %% get cluster level percentage through time
        % 1 for the most positive going cluster
        make_plots = 'yes';
        if cfg.tail == 1
            xlim = 256;
            title = 'Most positive going cluster through time as a % of entire volume';
            calculate_cluster_size(stat, 1, make_plots, title, xlim, 'positive', ...
                save_path);
        elseif cfg.tail == 0
            xlim = 256;
            title = 'Most positive going cluster through time as a % of entire volume';
            calculate_cluster_size(stat, 1, make_plots, title, xlim, 'positive', ...
                save_path);
            title = 'Most negative going cluster through time as a % of entire volume';
            calculate_cluster_size(stat, 1, make_plots, title, xlim, 'negative', ...
                save_path);
        end

        %% make pretty plots
        %cd('C:\Users\CDoga\OneDrive\Documents\\PhD\fieldtrip');
        alpha = 0.05;

        if cfg.tail == 1
            create_viz_topographic_maps(data, stat, start_latency, end_latency, ...
                alpha, 'positive', save_path)
        elseif cfg.tail == 0
            create_viz_topographic_maps(data, stat, start_latency, end_latency, ...
                alpha, 'positive', save_path)
            create_viz_topographic_maps(data, stat, start_latency, end_latency, ...
                alpha, 'negative', save_path)
        end
    end
end

%% plot and save the design matrix
function save_desgin_matrix(design_matrix, n_participants, save_path, experiment_type)
    plot(design_matrix(1:n_participants), 'color', 'r', 'LineWidth', 1);
    hold on;
    plot(design_matrix(n_participants+1:n_participants*2), 'color', 'g', 'LineWidth', 1);
    plot(design_matrix((n_participants*2)+1:n_participants*3), 'color', 'b', 'LineWidth', 1);
    xlabel('Participants');
    ylabel('Interaction');
    if strcmp(experiment_type, 'habituation')
        legend({'P1', 'P2', 'P3'},'Location','northwest')
    else
        legend({'Onsets 2:3', 'Onsets 4:5', 'Onsets 6:7'},'Location','northwest')
    end
        set(gcf,'Position',[100 100 1000 1000])
    save_dir = strcat(save_path, '\', 'design_matrix.png');
    exportgraphics(gcf,save_dir,'Resolution',500);
    close;   
end

%% generate ERPs
function generate_peak_erps(master_dir, main_path, experiment_type, ...
    stat, peak_information, effect_type, regression_type, desired_cluster, ...
    save_dir, bootstrap_ci_erps)
    
    df = stat.df;
    time = stat.time;
    peak_electrode = peak_information.electrode;
    peak_time = peak_information.time;
    peak_t_value = peak_information.t_value;
    
    if strcmp(effect_type, 'positive')
        if numel(stat.posclusters) < 1 || strcmp(peak_electrode, '')
           return ; 
        end
        
        labels = stat.posclusterslabelmat;
        labels(labels>desired_cluster) =0;
        pvalue = round(stat.posclusters(desired_cluster).prob,5);
        cluster_size = stat.posclusters(desired_cluster).clusterstat;
        plot_desc = 'positive_peak_erp.png';
    else
        if numel(stat.negclusters) < 1 || strcmp(peak_electrode, '')
            return ; 
        end
        
        
        labels = stat.negclusterslabelmat;
        peak_t_value = abs(peak_t_value);
        labels(labels>desired_cluster) =0;
        pvalue = round(stat.negclusters(desired_cluster).prob,4);
        cluster_size = stat.negclusters(desired_cluster).clusterstat;
        plot_desc = 'negative_peak_erp.png';
    end

    through_time = sum(labels);
    start_idx = find(through_time(:), desired_cluster, 'first');
    end_idx = find(through_time(:), desired_cluster, 'last');
    start_of_effect = time(start_idx);
    end_of_effect = time(end_idx);
    
    save_dir = strcat(save_dir, '\', plot_desc);
    
    generate_plots(master_dir, main_path, experiment_type, start_of_effect,...
        end_of_effect, peak_electrode, peak_time, peak_t_value, df, ...
        regression_type, pvalue, cluster_size, save_dir, effect_type, bootstrap_ci_erps)
    
    close;
end

%% Create a hacked ROI in the data
function new_data = create_hacked_roi(data, roi, weight_roi)
    roi_clusters = roi.clusters;
    roi_time = roi.time;
    
    if weight_roi == 1
        [electrodes, ~] = size(roi_clusters);
        total_sum = sum(roi_clusters);
        weighted_template = total_sum/electrodes;
    end

    start_latency = NaN;
    end_latency = NaN;
    new_data = {};
    for idx_i = 1:numel(data)
        each_participant = data{idx_i};
        participant_data = each_participant.avg;
        participant_time = each_participant.time;
        [electrodes, time] = size(participant_data);
        new_participant_data = NaN(electrodes, time);
        time_x = 1:1:time;
        time_x = time_x * 2;
     
        for roi_idx = 1:numel(roi_time)
            t = roi_time(roi_idx)*1000;
            [~,idx]=min(abs(time_x-t));
            idx = idx + 100; % for baselining period
            clusters_at_t = roi_clusters(:,roi_idx);
            
            if isnan(start_latency) && sum(clusters_at_t)>=1
                start_latency =t;
            elseif roi_idx == numel(roi_time)
                end_latency = t; 
            end
            
            for electrode_idx=1:numel(clusters_at_t)
                if clusters_at_t(electrode_idx) == 1
                    new_participant_data(electrode_idx, idx) = participant_data(electrode_idx, idx);
                    if weight_roi == 1
                        weighting_at_t =  weighted_template(roi_idx);
                        new_participant_data(electrode_idx, idx) = new_participant_data(electrode_idx, idx) * weighting_at_t;
                    end      
                end
            end
            
        end
        

        
        
        %[~,start_idx]=min(abs(time_x-start_latency));
        %[~,end_idx]=min(abs(time_x-end_latency));
        %new_participant_data = new_participant_data(:,start_idx:end_idx);
        each_participant.avg = new_participant_data;  
        
        %time = each_participant.time;
        %[~,start_idx]=min(abs(time-start_latency/1000));
        %[~,end_idx]=min(abs(time-end_latency/1000));
        
        new_data{idx_i} = each_participant;
    end
end

%% baed on onsets 2-8 mean intercept effect, get the start and end latency
%% based on our ROI
function get_start_and_end_latency_of_mean_intercept(stat)
    load(roi_fn);
    roi_labels = roi.label;
    stat_labels = stat.label;
    time = stat.time;
    cluster_labels = stat.posclusterslabelmat;
    cluster_labels(cluster_labels>1) = 0; 
    cluster_sum_by_time = sum(cluster_labels);
    
    start_found = 0;
    end_found=0;
    start_latecy = 0;
    end_latency = 0;
    
    for i = 1:numel(cluster_sum_by_time)
       if cluster_sum_by_time(i) > 0
          electrodes =  cluster_labels(:,i);
          electrodes_with_effect = find(electrodes==1);
          t = time(i);
            
          for electrode_idx = 1:numel(stat_labels)
              if any(electrodes_with_effect(:)==electrode_idx)
                electrode_name = stat_labels(electrode_idx);
                if numel(intersect(electrode_name, roi_labels)) ==1
                   if start_found == 0
                      start_latency =  t;
                      start_found =1;
                   end          
                end
              end
          end
       end
    end
end

%% based on onsets 2-8 mean intercept effect, save the electrodes 
% which are at the peak level - used for ROI analysis
% saves the neighbourhood of electrodes for use later on
function get_region_of_interest_electrodes(stat, desired_cluster, experiment_type, roi_applied)
    roi_clusters = stat.posclusterslabelmat;
    roi_clusters(roi_clusters>desired_cluster) = 0;
    time = stat.time;
    roi.clusters = roi_clusters;
    roi.time = time;
    
    if contains(experiment_type, 'onsets-2-8')  && contains(roi_applied, 'one-tailed')
        save 'C:\Users\CDoga\OneDrive\Documents\PhD\fieldtrip\one_tailed_roi_28.mat' roi; 
    elseif contains(experiment_type, 'onsets-1') && contains(roi_applied, 'one-tailed')
        save 'C:\Users\CDoga\OneDrive\Documents\PhD\fieldtrip\one_tailed_roi_1.mat' roi; 
    elseif contains(experiment_type, 'onsets-2-8')  && contains(roi_applied, 'two-tailed')
        save 'C:\ProgramFiles\PhD\fieldtrip\roi\two_tailed_roi_28.mat' roi; 
    elseif contains(experiment_type, 'onsets-1') && contains(roi_applied, 'two-tailed')
        save 'C:\ProgramFiles\PhD\fieldtrip\roi\two_tailed_roi_1.mat' roi; 
    end
   
end

    %% plots the cluster effect through time
function create_viz_cluster_effect(stat, alpha)
    cfg = [];
    cfg.alpha = alpha;
    cfg.parameter = 'stat';
    cfg.zlim = [-4 4];
    cfg.xlim = [0.56, 0.256];
    cfg.highlightcolorpos = [1,0,0]; 
    cfg.highlightsymbolseries =  ['*', '*','*','*','*'];
    %cfg.toi = [0.56, 0.256];
    ft_clusterplot(cfg, stat);
end

%% used to calculate the cluster size through time
function calculate_cluster_size(stat, desired_cluster, make_plots, ptitle, xlim_t, type, save_dir)
    if contains(type, 'positive')
        cluster_labelling_mtx = stat.posclusterslabelmat;
        save_dir = strcat(save_dir, '\', 'positive_cluster.png');
    elseif contains(type, 'negative')
        cluster_labelling_mtx = stat.negclusterslabelmat;
        save_dir = strcat(save_dir, '\', 'negative_cluster.png');
    end
    
    time_mtx = stat.time;
    
    [electrodes, time] = size(cluster_labelling_mtx);
    
    cluster_size_through_time = zeros(2,time);
    
    for i = 1:time
        t = time_mtx(i);
        electrodes_in_time = cluster_labelling_mtx(:,i);
        clusters_in_time = find(electrodes_in_time==desired_cluster);
        cluster_size_through_time(1,i) = t;
        cluster_size_through_time(2,i) = numel(clusters_in_time)/electrodes;
    end
    
    if make_plots == 'yes'
        area(cluster_size_through_time(1,:)*1000, cluster_size_through_time(2,:)*100);
        ylim([0 100]);
        grid on;
        xlabel('Time (ms)');
        ylabel('Percentage of cluster');
        %xlim([0,260])
        xlim([0,xlim_t])
        title(ptitle, 'FontSize', 14); 
    end
   
    set(gcf,'Position',[100 100 1000 1000])
    exportgraphics(gcf,save_dir,'Resolution',500);
    close;
end

%% used to generate 'peak' level stats
function [peak_level_electrode, all_electrode_stats] = get_peak_level_stats(stat, cluster, type)
    peak_level_stats = containers.Map;

    time = stat.time;
    electrodes = stat.label;
    t_values = stat.stat;
    if contains(type, 'positive')
        cluster_matrix_locations = stat.posclusterslabelmat;
    elseif contains(type, 'negative')
        cluster_matrix_locations = stat.negclusterslabelmat;
    end
    
    [rows, columns] = size(cluster_matrix_locations);
    
    for col = 1:columns
        for row = 1:rows
            
            t_value = t_values(row, col);
            t = time(col);
            electrode = electrodes{row};

            which_cluster = cluster_matrix_locations(row, col);
            if which_cluster == cluster
                keys = peak_level_stats.keys;
                if any(strcmp(keys,electrode))
                    previous_values = peak_level_stats(electrode);
                    previous_t = previous_values{2};
                    if t_value > previous_t && contains(type, 'positive')
                        peak_level_stats(electrode) = {t, t_value};
                    elseif t_value < previous_t && contains(type, 'negative')
                        peak_level_stats(electrode) = {t, t_value};
                    end
                else
                    peak_level_stats(electrode) = {0, 0};
                end
            end
        end 
    end
    peak_level_electrode = get_most_significant_electrode(peak_level_stats, type);
    all_electrode_stats = get_all_electrode_stats(peak_level_stats, type);
end

function mtx = get_all_electrode_stats(peak_level_stats, type)
    keys = peak_level_stats.keys;
    electrodes = cell(numel(keys), 1);
    time = zeros(numel(keys), 1);
    t_value = zeros(numel(keys), 1);
    
    for i = 1:numel(keys)
        electrode = keys{i};
        stats = peak_level_stats(electrode);
        tv = stats{2};
        t = stats{1};
        
        electrodes{i} = electrode;
        time(i) = t;
        t_value(i) = tv;
    end
    
    mtx = table(electrodes, time, t_value);
    if strcmp(type, 'positive')
        mtx = sortrows(mtx, {'t_value'}, {'descend'});
    elseif strcmp(type, 'negative')
        mtx = sortrows(mtx, {'t_value'}, {'ascend'});
    end
end

function most_significant = get_most_significant_electrode(peak_level_stats, type)
    keys = peak_level_stats.keys;
    
    most_significant.electrode = '';
    most_significant.time = 0;
    most_significant.t_value = 0;
    for i = 1:numel(keys)
        electrode = keys{i};
        stats = peak_level_stats(electrode);
        t_value = stats{2};
        time = stats{1};
        if t_value > most_significant.t_value && contains(type, 'positive')
            most_significant.electrode = electrode;
            most_significant.time = time;
            most_significant.t_value = t_value;
        elseif t_value < most_significant.t_value && contains(type, 'negative')
            most_significant.electrode = electrode;
            most_significant.time = time;
            most_significant.t_value = t_value; 
        end
    end
end
%% used to create the topographic maps
function create_viz_topographic_maps(data1, stat, start_latency, ... 
    end_latency, alpha, type, save_dir)

    if numel(stat.negclusters) < 1 && strcmp(type, 'negative')
        return;
    elseif numel(stat.posclusters) < 1 && strcmp(type,'positive')
        return;
    end

    % fixes the interpolation issue of NaNs
    stat.stat(isnan(stat.stat))=0;
        
    
    cfg = [];
    cfg.channel   = 'all';
    cfg.latency   = 'all';
    cfg.parameter = 'avg';
    
    grand_avg1 = ft_timelockgrandaverage(cfg, data1{:});
    grand_avg1.elec = data1{1}.elec;
    [i1,i2] = match_str(grand_avg1.label, stat.label);
    
    figure;
    timestep = 0.01; % in seconds usually 0.01
    sampling_rate = 512;
    sample_count  = length(stat.time);
    
    j = [start_latency:timestep:end_latency]; % Temporal endpoints (in seconds) of the ERP average computed in each subplot
    m = [1:timestep*sampling_rate:sample_count];  % temporal endpoints in M/EEG samples
    
    if contains(type, 'positive')
        pos_cluster_pvals = [stat.posclusters(:).prob];
        pos_clust = find(pos_cluster_pvals < alpha);
        clust = ismember(stat.posclusterslabelmat, pos_clust);
        save_dir = strcat(save_dir, '\', 'positive_topographic.png');
        t = 'Positive Tail: Most Postiive Going Cluster';
    elseif contains(type, 'negative')
        neg_cluster_pvals = [stat.negclusters(:).prob];
        neg_clust = find(neg_cluster_pvals < alpha);
        clust = ismember(stat.negclusterslabelmat, neg_clust);    
        save_dir = strcat(save_dir, '\', 'negative_topographic.png');
        t = 'Negative Tail: Most Negative Going Cluster';
    end
    max_iter = numel(m)-1;
    
    for k = 1:max_iter
         subplot(4,5,k);
         cfg = [];
         cfg.xlim = [j(k) j(k+1)];
         cfg.zlim = [-5e-14 5e-14];
         pos_int = zeros(numel(grand_avg1.label),1);
         pos_int(i1) = all(clust(i2, m(k):m(k+1)), 2);
         cfg.highlight = 'on';
         cfg.highlightchannel = find(pos_int);
         cfg.highlightcolor = {'r',[0 0 1]};
         cfg.comment = 'xlim';
         cfg.commentpos = 'title';
         cfg.parameter = 'stat';
         cfg.zlim=[-4,4];
         %cfg.colorbar = 'SouthOutside';
         %cfg.layout = 'biosemi128.lay';
         ft_topoplotER(cfg, stat);
         %ft_clusterplot(cfg, stat)
    end
    
    %%cfg = [];
    %%cfg.alpha  = 0.025;
    %%cfg.parameter = 'stat';
    %%cfg.zlim   = [-4 4];
    %%ft_clusterplot(cfg, stat);
    title(t);
    set(gcf,'Position',[100 100 1000 1000])
    exportgraphics(gcf,save_dir,'Resolution',500);
    close;
end

%% set all my values to 0 to hack a T test
function data = set_values_to_zero(data)
    for idx = 1:numel(data)
        participant_data = data{idx};
        series = participant_data.avg;
        series(:) = 0;
        participant_data.avg = series;
        data{idx} = participant_data;
    end
end
%% function to create the design matrix for all experiments
function [design, new_participants] = create_design_matrix_partitions(participant_order, participants, ...
    regression_type, partition, type_of_effect)

    scores = return_scores(regression_type, type_of_effect);
    
    if strcmp(regression_type, 'headache-scores-entire-median-split') ...
            || strcmp(regression_type, 'discomfort-scores-entire-median-split') ...
            || strcmp(regression_type, 'aura-scores-entire-median-split')
        score = scores.one;
        calc(:,1) = score(:,2);
        calc(:,2) = score(:,1);
        calc = sortrows(calc);
        [rows, ~] = size(calc);
        calc(1:rows/2,1) = -1;
        calc(((rows+1)/2):rows,1) =1;
        ratings(:,1) = calc(:,2);
        ratings(:,2) = calc(:,1);      
    else
        if partition == 1
            ratings = scores.one;
        elseif partition == 2
            ratings = scores.two;
        elseif partition == 3
            ratings = scores.three;
        end
    end
    
    new_participants = {};
    cnt = 1;
    for j = 1:numel(participant_order)
        participant = participant_order(j);
        score = ratings(find(ratings(:,1)==participant),2);

        if ismember(participant, [24,25,27])
            continue;
        end
        
        if numel(score) > 0
            design(1,cnt)=score;
            new_participants{cnt} = participants{j};
            cnt = cnt + 1;
        end
    end
    
end

%% return scores
function scores = return_scores(regression_type, type_of_effect)
    if strcmp(regression_type, 'no-factor')
        scores.one = [
        1,1;2,1;3,1;4,1;5,1;6,1;7,1;8,1;9,1;10,1;11,1;12,1;13,1;14,1;16,1;
        17,1;20,1;21,1;22,1;23,1;26,1;28,1;29,1;30,1;31,1;32,1;33,1;34,1;38,1;
        39,1;
        ];

        scores.two = [
        1,1;2,1;3,1;4,1;5,1;6,1;7,1;8,1;9,1;10,1;11,1;12,1;13,1;14,1;16,1;
        17,1;20,1;21,1;22,1;23,1;26,1;28,1;29,1;30,1;31,1;32,1;33,1;34,1;38,1;
        39,1;
        ];
    
        scores.three = [
        1,1;2,1;3,1;4,1;5,1;6,1;7,1;8,1;9,1;10,1;11,1;12,1;13,1;14,1;16,1;
        17,1;20,1;21,1;22,1;23,1;26,1;28,1;29,1;30,1;31,1;32,1;33,1;34,1;38,1;
        39,1;
        ];
    elseif strcmp(regression_type, 'headache')
        dataset = [
        1,-0.2574;2,-0.0417;3,-0.6726;4,0.4236;5,1.781;6,-1.0608;7,-0.7657;
        8,0.1279;9,-0.6553;10,-0.2896;11,-0.5122;12,2.1424;13,-0.1803;
        14,1.4491;16,0.1157;17,-0.1649;20,-0.4721;21,1.0486;22,-0.554;23,-0.8912;
        24,-0.4481;25,-0.7581;26,-1.2784;28,0.2989;29,0.0439;30,-0.4732;31,-0.7701;
        32,-0.7037;33,-0.819;34,-0.7987;37,1.1507;38,-0.2806;39,0.8546;40,-0.3823;   
        ];
    
        scores.one = dataset;
        scores.two = dataset;
        scores.three = dataset;
    
        min_n = min(scores.one);
        scores.one(:,2) = scores.one(:,2) - min_n(2);
        scores.two(:,2) = scores.two(:,2) - min_n(2);
        scores.three(:,2) = scores.three(:,2) - min_n(2);
        
        if strcmp(type_of_effect, 'habituation')
            scores.one(:,2) = scores.one(:,2) * 2.72;
            scores.two(:,2) = scores.two(:,2) * 1.65;
            scores.three(:,2) = scores.three(:,2) * 1.00;
        elseif strcmp(type_of_effect, 'sensitization')
            scores.one(:,2) = scores.one(:,2) * 1.00;
            scores.two(:,2) = scores.two(:,2) * 1.50;
            scores.three(:,2) = scores.three(:,2) * 2.00;
        else
            error('Type of experiment not properly specified');
        end
                   
    elseif strcmp(regression_type, 'aura')

        min_n = min(scores.one);
        scores.one(:,2) = scores.one(:,2) - min_n(2);
        scores.two(:,2) = scores.two(:,2) - min_n(2);
        scores.three(:,2) = scores.three(:,2) - min_n(2);
    
        if strcmp(type_of_effect, 'habituation')
            scores.one(:,2) = scores.one(:,2) * 2.72;
            scores.two(:,2) = scores.two(:,2) * 1.65;
            scores.three(:,2) = scores.three(:,2) * 1.00;
        elseif strcmp(type_of_effect, 'sensitization')
            scores.one(:,2) = scores.one(:,2) * 1.00;
            scores.two(:,2) = scores.two(:,2) * 1.65;
            scores.three(:,2) = scores.three(:,2) * 2.72;
        else
            error('Type of experiment not properly specified');
        end

    elseif strcmp(regression_type, 'discomfort')


        min_n = min(scores.one);
        scores.one(:,2) = scores.one(:,2) - min_n(2);
        scores.two(:,2) = scores.two(:,2) - min_n(2);
        scores.three(:,2) = scores.three(:,2) - min_n(2);
    
    
        if strcmp(type_of_effect, 'habituation')
            scores.one(:,2) = scores.one(:,2) * 2.72;
            scores.two(:,2) = scores.two(:,2) * 1.65;
            scores.three(:,2) = scores.three(:,2) * 1.00;
        elseif strcmp(type_of_effect, 'sensitization')
            scores.one(:,2) = scores.one(:,2) * 1.00;
            scores.two(:,2) = scores.two(:,2) * 1.65;
            scores.three(:,2) = scores.three(:,2) * 2.72;
        else
            error('Type of experiment not properly specified');
        end

    elseif contains(regression_type, 'headache-scores-entire')
        scores.one = [
        1,-0.227;2,-0.052;3,-0.721;4,0.531;5,1.72;6,-1.176;7,-0.797;
        8,0.199;9,-0.692;10,-0.358;11,-0.585;12,2.041;13,-0.266;14,1.31;
        16,0.002;17,-0.15;20,-0.416;21,1.144;22,-0.565;23,-0.728;24,-0.435;
        25,-0.699;26,-1.35;28,0.363;29,0.042;30,-0.47;31,-0.704;32,-0.732;
        33,-0.881;34,-0.796;37,1.227;38,-0.337;39,1.077;40,-0.167
        ];
    
    elseif contains(regression_type, 'discomfort-scores-entire')
        scores.one = [
        1,-0.216;2,0.439;3,-0.495;4,1.848;5,-0.746;6,0.84;7,-0.659;8,0.987;
        9,-0.793;10,-0.919;11,0.359;12,-1.015;13,-0.758;14,1.368;16,1.022;
        17,1.562;20,0.21;21,0.098;22,0.647;23,-1.258;26,-0.531;28,-0.894;
        29,0.374;30,-0.593;31,2.267;32,-1.475;33,1.221;34,-0.736;38,-1.059;
        39,-0.728;  
        ];
       
    elseif contains(regression_type, 'aura-scores-entire')
        scores.one = [
        1,0.323;2,-0.109;3,-0.51;4,1.134;5,-0.639;6,-1.215;7,-0.33;8,0.752;
        9,-0.39;10,-0.722;11,-0.769;12,-1.063;13,-0.899;14,-1.467;16,-1.199;
        17,0.154;20,0.587;21,1.001;22,-0.117;23,1.721;26,-0.748;28,0.674;
        29,-0.024;30,0.036;31,0.7;32,-0.3;33,-0.65;34,0.026;38,-0.588;39,2.333;
        ];
    end
end

%% return the SPM data in a fieldtrip format
function [fieldtrip_data, participant_order] = from_fieldtrip_to_spm(n_participants, main_path, ...,
        filename, partition, required_type)
    fieldtrip_data = {};
    participant_order = [];
    cnt = 1;
    for participant = 1:n_participants
                
        participant_main_path = strcat(main_path, int2str(participant));
        if exist(participant_main_path, 'dir')
            if participant < 10
               p = strcat('0', int2str(participant));
            else
               p = int2str(participant);
            end

            % data structure
            data_structure = strcat('spmeeg_P', p);  
            data_structure = strcat(data_structure, '_075_80Hz.mat');
            data_structure = strcat(filename, data_structure); 
            cd(participant_main_path); 
            cd('SPM_ARCHIVE');
            
            if isfile(data_structure)
                load(data_structure);
            else
                disp(strcat('Cannot find participant',int2str(participant)));
                continue;
            end

            spm_eeg = meeg(D);
            fieldtrip_raw = spm_eeg.ftraw;
            n_trials = size(D.trials);
            n_trials = n_trials(2);

            for index_i = 1:n_trials
                label = D.trials(index_i).label;
                if partition.is_partition
                    if contains(label, partition.partition_number) && contains(label, 'medium')
                        med = fieldtrip_raw.trial(index_i);
                    elseif contains(label, partition.partition_number) && contains(label, 'thin')
                        thin = fieldtrip_raw.trial(index_i);
                    elseif contains(label, partition.partition_number) && contains(label, 'thick')
                        thick = fieldtrip_raw.trial(index_i);
                    end
                else
                     if contains(label, 'medium')
                        med = fieldtrip_raw.trial(index_i);
                    elseif contains(label, 'thin')
                        thin = fieldtrip_raw.trial(index_i);
                    elseif contains(label, 'thick')
                        thick = fieldtrip_raw.trial(index_i);                   
                     end
                end
            end

               
            % update the fieldtrip structure with fields of information
            if strcmp(required_type,'regression')
                fieldtrip_raw.avg = med{1} - (thin{1}+thick{1})/2;
                fieldtrip_raw.trialinfo = [1];
                fieldtrip_raw.time = fieldtrip_raw.time{1};
                fieldtrip_raw.dimord = 'chan_time';
                fieldtrip_raw = rmfield(fieldtrip_raw,'trial');
                fieldtrip_raw = remove_electrodes(fieldtrip_raw);
            elseif strcmp(required_type,'plotting')
                fieldtrip_raw.thin = thin{1};
                fieldtrip_raw.thick = thick{1};
                fieldtrip_raw.med = med{1};
                fieldtrip_raw.pgi = med{1} - (thin{1}+thick{1})/2;
                fieldtrip_raw.time = fieldtrip_raw.time{1};
            end
            
            % update object with fieldtrip data
            fieldtrip_data{cnt} = fieldtrip_raw;
            participant_order(cnt) = participant;
            cnt = cnt + 1;
        end
    end
end

function fieldtrip_raw = remove_electrodes(fieldtrip_raw)
    %to_remove = {'A11', 'A12', 'A13', 'A14', 'A24', 'A25', 'A26','A27', 'B8', 'B9','EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'HEOG', 'VEOG'};
    to_remove = {'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'HEOG', 'VEOG'};
    electorode_information = fieldtrip_raw.elec;
    
    new_chanpos = [];
    new_chantype = {};
    new_chanunit = {};
    new_elecpos = [];
    new_labels = {};
    
    cnt = 1;
    for idx = 1:numel(electorode_information.label)
        electrode =  electorode_information.label{idx};
        if ~ismember(electrode, to_remove)
            new_chanpos(cnt,:) = electorode_information.chanpos(idx,:);
            new_chantype{cnt} = electorode_information.chantype{idx};
            new_chanunit{cnt} = electorode_information.chanunit{idx};
            new_elecpos(cnt,:) = electorode_information.elecpos(idx,:);
            new_labels{cnt} = electorode_information.label{idx};
            cnt = cnt + 1;
        end  
    end
    
    
    fieldtrip_raw.elec.chanpos = new_chanpos;
    fieldtrip_raw.elec.chantype = new_chantype;
    fieldtrip_raw.elec.chanunit = new_chanunit;
    fieldtrip_raw.elec.elecpos = new_elecpos;
    fieldtrip_raw.elec.label = new_labels;
   
    cnt = 1;
    for idx = 1:numel(fieldtrip_raw.label)
        elec = fieldtrip_raw.label{idx};
        if ~ismember(elec, to_remove)
            new_elec{cnt} = elec;
            new_avg(cnt, :) = fieldtrip_raw.avg(idx, :);
            cnt = cnt+1;
        end
    end
    
    fieldtrip_raw.label = new_elec;
    fieldtrip_raw.avg = new_avg;
end

%% generate erp plots
function generate_plots(master_dir, main_path, experiment_type, start_peak, ...
    end_peak, peak_electrode, peak_effect, t_value, df, regression_type, ...
    pvalue, cluster_size, save_dir, effect_type, bootstrap_ci_erps)

    rmpath C:\ProgramFiles\spm8;
    addpath C:\ProgramFiles\spm12;
    cd(master_dir);

    %% Are we looking at onsets 2-8 or partitions
    % set up the experiment as needed
    if strcmp(experiment_type, 'onsets-2-8')
        data_file = 'averaged_onsets_2_3_4_5_6_7_8t1b1f1';
        n_participants = 40;
        start_latency = 0.056;
        end_latency = 0.256;

        partition.is_partition = 0;
        partition.partition_number = 0;

        [data, participant_order] = from_fieldtrip_to_spm(n_participants,main_path,data_file,partition, 'plotting');
        n_part = numel(data);
        design_matrix =  [1:n_part 1:n_part; ones(1,n_part) 2*ones(1,n_part)]; 

    elseif strcmp(experiment_type, 'onsets-1')
        data_file = 'averaged_onsets_1t1b1f1';
        n_participants = 40;

        partition.is_partition = 0;
        partition.partition_number = 0;

        [data, participant_order] = from_fieldtrip_to_spm(n_participants,main_path,data_file,partition, 'plotting');
        n_part = numel(data);
        design_matrix =  [1:n_part 1:n_part; ones(1,n_part) 2*ones(1,n_part)];
        
    elseif strcmp(experiment_type, 'Partitions (No Factor)') || strcmp(experiment_type, 'partitions-1') ||  strcmp(experiment_type, 'partitions-2-8')
        if contains(experiment_type, 'Partitions (No Factor)') || contains(experiment_type, 'partitions-2-8')
            data_file = 'grand-avg_trial-level_partitionsb1f1';
        elseif contains(experiment_type, 'partitions-1')
            data_file = 'grand-avg_trial-level_partitionsb1f1';
        end
        n_participants = 39;

        partition1.is_partition = 1; % partition 1
        partition1.partition_number = '1';
        partition2.is_partition = 1; % partition 2
        partition2.partition_number = '2';
        partition3.is_partition = 1; % partition 3
        partition3.partition_number = '3';

        [data1, participant_order_1] = from_fieldtrip_to_spm(n_participants,main_path,data_file,partition1, 'plotting');
        cd(master_dir);
        [data2, participant_order_2] = from_fieldtrip_to_spm(n_participants,main_path,data_file,partition2, 'plotting');
        cd(master_dir);
        [data3, participant_order_3] = from_fieldtrip_to_spm(n_participants,main_path,data_file,partition3, 'plotting');
        cd(master_dir);

        data = [data1,data2,data3];

        type_of_effect = 'habituation';

        [data1_h, data1_l] = get_partitions_medium_split(data1, participant_order_1,...
            regression_type, 1, type_of_effect);
        [data2_h, data2_l] = get_partitions_medium_split(data2, participant_order_2,...
            regression_type, 2, type_of_effect);
        [data3_h, data3_l] = get_partitions_medium_split(data3, participant_order_3,...
            regression_type, 3, type_of_effect);

    elseif strcmp(experiment_type, 'erps-23-45-67') || strcmp(experiment_type, 'erps-23-45-67-no-factor') 
        data_file23 = 'averaged_onsets_2_3t1b1f1';
        data_file45 = 'averaged_onsets_4_5t1b1f1';
        data_file67 = 'averaged_onsets_6_7t1b1f1';

        type_of_effect = 'sensitization';
        
        n_participants = 40;

        partition.is_partition = 0;
        partition.partition_number = 0;

        [data1, participant_order_1] = from_fieldtrip_to_spm(n_participants,main_path,data_file23,partition, 'plotting');
        [data1_h, data1_l] = get_partitions_medium_split(data1, participant_order_1,...
            regression_type, 1, type_of_effect);

        [data2, participant_order_1] = from_fieldtrip_to_spm(n_participants,main_path,data_file45,partition, 'plotting');
        [data2_h, data2_l] = get_partitions_medium_split(data2, participant_order_1,...
            regression_type, 1, type_of_effect);

        [data3, participant_order_1] = from_fieldtrip_to_spm(n_participants,main_path,data_file67,partition, 'plotting');
        [data3_h, data3_l] = get_partitions_medium_split(data3, participant_order_1,...
            regression_type, 1,  type_of_effect);
        
        data = [data1, data2, data3];
    end
    %% generate_supplementary information and indices used to plot
    if strcmp(experiment_type, 'partitions-2-8')
        experiment_name = 'Partitions (2:8) ';
    elseif strcmp(experiment_type, 'erps-23-45-67')
        experiment_name = 'Onsets (2,3; 4,5; 6,7)';
    elseif strcmp(experiment_type, 'erps-23-45-67-no-factor')
        experiment_name = 'ERPs 2,3; 4,5; 6,7 (No Factor)';
    else
       experiment_name = experiment_type;
    end
    
    
    regression_type = regexprep(regression_type,'(\<[a-z])','${upper($1)}');
    effect_type = strcat(regexprep(effect_type,'(\<[a-z])','${upper($1)}'), ' Tail');
    mtitle = strcat(effect_type, {' '}, experiment_name, {' ' }, regression_type, {' '}, peak_electrode);
    mtitle = mtitle{1};
    
    electrode_idx = find(contains(data{1}.label,peak_electrode));
    start_peak = start_peak*1000;
    end_peak = end_peak*1000;
    cohens_d = round((2*t_value)/sqrt(df),2);
    effect_size = round(sqrt((t_value*t_value)/((t_value*t_value)+df)),2);
    
    time = data{1}.time * 1000;
    peak_effect = peak_effect*1000;
    [~, peak_effect_idx] = min(abs(time-peak_effect));
    t_value = round(t_value, 2);
    cluster_size = round(cluster_size, 0);
    
    t = tiledlayout(5,2, 'TileSpacing','Compact');

    if contains(experiment_type, 'onsets-2-8')
       time = data{1}.time * 1000;
       avg_data = calculate_grand_averages(data);
       nexttile
       plot(time, avg_data.pgi(electrode_idx,:), 'color', 'r')
       xlim([-100, 300])
       ylim([-2, 8])
       grid on
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'PGI'},'Location','northeast')

       nexttile
       hold on;
       plot(time, avg_data.thin(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, avg_data.med(electrode_idx,:), 'color', 'b','LineWidth', 1)
       plot(time, avg_data.thick(electrode_idx,:), 'color', 'k','LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       xlim([-100, 300])
       ylim([-2, 8])
       grid on
       legend({'Thin','Medium','Thick'},'Location','northeast')

       hold off;
    elseif contains(experiment_type, 'partitions-2-8') && ~strcmp(regression_type, 'No-Factor')
       time = data{1}.time * 1000;

       avg_1_h  = calculate_grand_averages(data1_h);
       avg_2_h  = calculate_grand_averages(data2_h);
       avg_3_h  = calculate_grand_averages(data3_h);

       avg_1_l  = calculate_grand_averages(data1_l);
       avg_2_l  = calculate_grand_averages(data2_l);
       avg_3_l  = calculate_grand_averages(data3_l);

       % pgi
       nexttile
       hold on;
       plot(time, avg_1_h.pgi(electrode_idx,:), 'color', 'r','LineWidth', 1)
       plot(time, avg_2_h.pgi(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, avg_3_h.pgi(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'P1', 'P2', 'P3'},'Location','northwest')
       title('High Group: Partitions: PGI');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;

       nexttile
       hold on;
       plot(time, avg_1_l.pgi(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
       plot(time, avg_2_l.pgi(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, avg_3_l.pgi(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'P1', 'P2', 'P3'},'Location','northwest')
       title('Low Group: Partitions: PGI');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;
       
       % medium high group --
       % pgi
       nexttile
       hold on;
       plot(time, avg_1_h.med(electrode_idx,:), 'color', 'r','LineWidth', 1)
       plot(time, avg_2_h.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, avg_3_h.med(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'Med-P1', 'Med-P2', 'Med-P3'},'Location','northwest')
       title('High Group: Medium Through the Partitions');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;
       
       % medium low group --
       nexttile
       hold on;
       plot(time, avg_1_l.med(electrode_idx,:), 'color', 'r','LineWidth', 1)
       plot(time, avg_2_l.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, avg_3_l.med(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'Med-P1', 'Med-P2', 'Med-P3'},'Location','northwest')
       title('Low Group: Medium Through the Partitions');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;
       
       % p1
       nexttile
       hold on;
       plot(time, avg_1_h.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
       plot(time, avg_1_h.med(electrode_idx,:), 'color', 'g','LineWidth', 1)
       plot(time, avg_1_h.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
       title('High Group: Partition 1');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;

       nexttile
       hold on;
       plot(time, avg_1_l.thin(electrode_idx,:), 'color', 'r','LineWidth', 1)
       plot(time, avg_1_l.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, avg_1_l.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
       title('Low Group: Partition 1');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;

       % p2
       nexttile
       hold on;
       plot(time, avg_2_h.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
       plot(time, avg_2_h.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, avg_2_h.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
       title('High Group: Partition 2');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;

       nexttile
       hold on;
       plot(time, avg_2_l.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
       plot(time, avg_2_l.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, avg_2_l.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
       title('Low Group: Partition 2');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;

       % p3
       nexttile
       hold on;
       plot(time, avg_3_h.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
       plot(time, avg_3_h.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, avg_3_h.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
       title('High Group: Partition 3');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;

       nexttile
       hold on;
       plot(time, avg_3_l.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
       plot(time, avg_3_l.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, avg_3_l.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
       title('Low Group: Partition 3');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;
    elseif contains(experiment_type, 'partitions-2-8') && strcmp(regression_type, 'No-Factor')
        time = data1{1}.time * 1000;    

        t = tiledlayout(5,1, 'TileSpacing','Compact');
        
        grnd_avg_1 = calculate_grand_averages(data1);
        grnd_avg_2 = calculate_grand_averages(data2);
        grnd_avg_3 = calculate_grand_averages(data3);

       % pgi
        nexttile
        hold on;
        plot(time, grnd_avg_1.pgi(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_2.pgi(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_3.pgi(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'PGI: Partition 1', 'PGI: Partition 2', 'PGI: Partition 3'},'Location','northwest')
        title('Pattern Glare Index');
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

        
       % medium high group --
       % pgi
       nexttile
       hold on;
       plot(time, grnd_avg_1.med(electrode_idx,:), 'color', 'r','LineWidth', 1)
       plot(time, grnd_avg_2.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, grnd_avg_3.med(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'Med-P1', 'Med-P2', 'Med-P3'},'Location','northwest')
       title('Medium Through the Partitions');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;
        
           % pgi
        nexttile
        hold on;
        plot(time, grnd_avg_1.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_1.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_1.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
        title('Partition 1');
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

               % pgi
        nexttile
        hold on;
        plot(time, grnd_avg_2.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_2.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_2.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
        title('Partition 2');
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

               % pgi
        nexttile
        hold on;
        plot(time, grnd_avg_3.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_3.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_3.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
        title('Partition 3');
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

       xlim([-100, 300])
       ylim([-6, 12])
       grid on;
       hold off;    
    elseif strcmp(experiment_type, 'erps-23-45-67') && ~strcmp(regression_type, 'No-Factor')
        t = tiledlayout(5,2, 'TileSpacing','Compact');
        time = data1{1}.time * 1000;    

        grnd_avg_1_h = calculate_grand_averages(data1_h);
        grnd_avg_1_l = calculate_grand_averages(data1_l);

        grnd_avg_2_h = calculate_grand_averages(data2_h);
        grnd_avg_2_l = calculate_grand_averages(data2_l);

        grnd_avg_3_h = calculate_grand_averages(data3_h);
        grnd_avg_3_l = calculate_grand_averages(data3_l);

        % pgi - high group
        nexttile
        hold on;
        plot(time, grnd_avg_1_h.pgi(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_2_h.pgi(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_3_h.pgi(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Onsets 2:3 PGI', 'Onsets 4:5 PGI', 'Onsets 6:7 PGI'},'Location','northwest')
        title('Pattern Glare Index - High Group');

        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

        % pgi low group
        nexttile
        hold on;
        plot(time, grnd_avg_1_l.pgi(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_2_l.pgi(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_3_l.pgi(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Onsets 2:3 PGI', 'Onsets 4:5 PGI', 'Onsets 6:7 PGI'},'Location','northwest')
        title('Pattern Glare Index - Low Group');
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

       % medium high group --
       % pgi
       nexttile
       hold on;
       plot(time, grnd_avg_1_h.med(electrode_idx,:), 'color', 'r','LineWidth', 1)
       plot(time, grnd_avg_2_h.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, grnd_avg_3_h.med(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'Med-2:3', 'Med-4:5', 'Med-6:7'},'Location','northwest')
       title('High Group: Medium Through the Onsets');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;
       
       % medium low group --
       nexttile
       hold on;
       plot(time, grnd_avg_1_l.med(electrode_idx,:), 'color', 'r','LineWidth', 1)
       plot(time, grnd_avg_2_l.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
       plot(time, grnd_avg_3_l.med(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
       xline(start_peak, '-');
       xline(end_peak, '-');
       xline(peak_effect, '--r');
       legend({'Med-2:3', 'Med-4:5', 'Med-6:7'},'Location','northwest')
       title('Low Group: Medium Through the Onsets');
       xlim([-100, 300])
       ylim([-5, 12])
       grid on;
       hold off;        

        % onsets 2,3 - high group
        nexttile
        hold on;
        plot(time, grnd_avg_1_h.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_1_h.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_1_h.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
        title('Onsets 2,3: High Group')
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

        % onsets 2,3 - low group
        nexttile
        hold on;
        plot(time, grnd_avg_1_l.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_1_l.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_1_l.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
        title('Onsets 2,3: Low Group');
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

        % onsets 4,5 - high group
        nexttile
        hold on;
        plot(time, grnd_avg_2_h.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_2_h.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_2_h.thick(electrode_idx,:), 'color', 'b','LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
        title('Onsets 4,5: High Group')
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

        % onsets 4,5 - low group
        nexttile
        hold on;
        plot(time, grnd_avg_2_l.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_2_l.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_2_l.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
        title('Onsets 4,5: Low Group');
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

        % onsets 6,7 - high group
        nexttile
        hold on;
        plot(time, grnd_avg_3_h.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_3_h.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_3_h.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
        title('Onsets 6,7: High Group')
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

        % onsets 6,7 - low group
        nexttile
        hold on;
        plot(time, grnd_avg_3_l.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_3_l.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_3_l.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
        title('Onsets 6,7: Low Group');
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;
    elseif strcmp(experiment_type, 'erps-23-45-67') && strcmp(regression_type, 'No-Factor')
        t = tiledlayout(5,1, 'TileSpacing','Compact');
        time = data1{1}.time * 1000;       

        grnd_avg_1_h = calculate_grand_averages(data1_h);
        grnd_avg_1_l = calculate_grand_averages(data1_l);
        grnd_avg_1_all.pgi = (grnd_avg_1_l.pgi + grnd_avg_1_h.pgi)/2;
        grnd_avg_1_all.thin = (grnd_avg_1_l.thin + grnd_avg_1_h.thin)/2;
        grnd_avg_1_all.med = (grnd_avg_1_l.med + grnd_avg_1_h.med)/2;
        grnd_avg_1_all.thick = (grnd_avg_1_l.thick + grnd_avg_1_h.thick)/2;

        grnd_avg_2_h = calculate_grand_averages(data2_h);
        grnd_avg_2_l = calculate_grand_averages(data2_l);
        grnd_avg_2_all.pgi = (grnd_avg_2_l.pgi + grnd_avg_2_h.pgi)/2;
        grnd_avg_2_all.thin = (grnd_avg_2_l.thin + grnd_avg_2_h.thin)/2;
        grnd_avg_2_all.thick = (grnd_avg_2_l.thick + grnd_avg_2_h.thick)/2;
        grnd_avg_2_all.med = (grnd_avg_2_l.med + grnd_avg_2_h.med)/2;

        grnd_avg_3_h = calculate_grand_averages(data3_h);
        grnd_avg_3_l = calculate_grand_averages(data3_l);
        grnd_avg_3_all.pgi = (grnd_avg_3_l.pgi + grnd_avg_3_h.pgi)/2;
        grnd_avg_3_all.thin = (grnd_avg_3_l.thin + grnd_avg_3_h.thin)/2;
        grnd_avg_3_all.thick = (grnd_avg_3_l.thick + grnd_avg_3_h.thick)/2;
        grnd_avg_3_all.med = (grnd_avg_3_l.med + grnd_avg_3_h.med)/2;   

        % pgi - high group
        nexttile
        hold on;
        plot(time, grnd_avg_1_all.pgi(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_2_all.pgi(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_3_all.pgi(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Onsets 2;3', 'Onsets 4;5', 'Onsets 6;7'},'Location','northwest')
        title('Pattern Glare Index');
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

       % medium low group --
        nexttile
        hold on;
        plot(time, grnd_avg_1_all.med(electrode_idx,:), 'color', 'r','LineWidth', 1)
        plot(time, grnd_avg_2_all.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_3_all.med(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Med-2:3', 'Med-4:5', 'Med-6:7'},'Location','northwest')
        title('Medium Through the Onsets');
        xlim([-100, 300])
        ylim([-5, 12])
        grid on;
        hold off;   
        
        % onsets 2,3 - high group
        nexttile
        hold on;
        plot(time, grnd_avg_1_all.thin(electrode_idx,:), 'color', 'r','LineWidth', 1)
        plot(time, grnd_avg_1_all.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_1_all.thick(electrode_idx,:), 'color', 'b','LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
        title('Onsets 2,3')
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

        % onsets 4,5 - high group
        nexttile
        hold on;
        plot(time, grnd_avg_2_all.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_2_all.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_2_all.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
        title('Onsets 4,5')
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

        % onsets 6,7 - high group
        nexttile
        hold on;
        plot(time, grnd_avg_3_all.thin(electrode_idx,:), 'color', 'r', 'LineWidth', 1)
        plot(time, grnd_avg_3_all.med(electrode_idx,:), 'color', 'g', 'LineWidth', 1)
        plot(time, grnd_avg_3_all.thick(electrode_idx,:), 'color', 'b', 'LineWidth', 1)
        xline(start_peak, '-');
        xline(end_peak, '-');
        xline(peak_effect, '--r');
        legend({'Thin', 'Medium', 'Thick'},'Location','northwest')
        title('Onsets 6,7')
        xlim([-100, 300])
        ylim([-6, 12])
        grid on;
        hold off;

            
    end
    title(t, mtitle, 'FontSize', 14);
    effect_peak = strcat('Effect Size Correlation: ', {' '}, num2str(effect_size), ...
        ' Cohens D: ', {' '}, num2str(cohens_d), ' T: ', {' '}, num2str(t_value));
    effect_peak = effect_peak{1};
    cluster_level = strcat('P-value:', {' '}, num2str(pvalue), ' Cluster Size', ...
        {' '}, num2str(cluster_size));
    cluster_level = cluster_level{1};
    subtitle(t, {effect_peak, cluster_level}, 'FontSize', 12)
    set(gcf,'Position',[100 100 1000 1000])
    exportgraphics(gcf,save_dir,'Resolution',500);
end
%% calculate partitions splits
function [data_high, data_low] = get_partitions_medium_split(data, participant_order, regression_type, partition, type_of_effect)
    function split_data = get_participants(data, all_ids, current_ids)
        cnt = 1;
        split_data = {};
        for i=1:numel(data)
            participant = data{i};
            id = all_ids(i);

            if ismember(id, current_ids)
                split_data{cnt} = participant;
                cnt = cnt+1;
            end     
        end      
    end    

    scores = return_scores(regression_type, type_of_effect);

    if contains(regression_type, 'median-split')
        ratings = scores.one;
    else
        if partition == 1
            ratings = scores.one;
        elseif partition == 2
            ratings = scores.two;
        elseif partition == 3
            ratings = scores.three;
        end
    end

    if strcmp(regression_type, 'headache-masked')
        ps = ratings(:,1);
        scores = ratings(:,2);
        scores(scores>0)=1;
        scores(scores<0)=-1;
        ratings(:,1) = ps;
        ratings(:,2) = scores;
        low_ids = ratings(ratings(:,2)<0);
        high_ids = ratings(ratings(:,2)>0);
    elseif strcmp(regression_type, 'headache-quartile')
        sorted(:,1) = ratings(:,2);
        sorted(:,2) = ratings(:,1);
        sorted = flipud(sortrows(sorted));
        n = ceil(numel(sorted(:,1))/2);
        high = sorted(1:n/2,:);
        high_ids = high(:,2);
        low = sorted(n+n/2+1:end,:);
        low_ids = low(:,2);
    else
        sorted(:,1) = ratings(:,2);
        sorted(:,2) = ratings(:,1);
        sorted = flipud(sortrows(sorted));
        n = ceil(numel(sorted(:,1))/2);
        high = sorted(1:n,:);
        high_ids = high(:,2);
        low = sorted(n+1:end,:);
        low_ids = low(:,2);
    end
    data_high = get_participants(data, participant_order, high_ids);
    data_low = get_participants(data, participant_order, low_ids);
end

    %% calculate grand avg
function avg_data = calculate_grand_averages(data)
    avg_data = {};

    function avg_mtx = average_matricies(data, type)
        matricies = [];
        cnt = 1;
        for d=data
            if contains(type,'thin')
                data = d{1}.thin;
            elseif contains(type, 'med')
                data = d{1}.med;
            elseif contains(type, 'thick')
                data = d{1}.thick;
            elseif contains(type, 'pgi')
                data = d{1}.pgi;
            end
            matricies(:,:,cnt) = data;
            cnt = cnt+1;
        end
        avg_mtx = mean(matricies, 3);
    end

    avg_data.thin = average_matricies(data, 'thin');
    avg_data.thick = average_matricies(data, 'thick');
    avg_data.med = average_matricies(data, 'med');
    avg_data.pgi = average_matricies(data, 'pgi');
end

function erp_with_cis = bootstrap_erps(erp, electrode_idx)
    function combine_data(erp, type)
        disp('hello');
    end

    % get all the data for each condition
    thick = combine_data(erp, 'thick');
    thin = combine_data(erp, 'thin');
    medium = combine_data(erp, 'medium');
end