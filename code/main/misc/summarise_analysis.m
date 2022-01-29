super_dir = "D:\PhD\results\trial_level_2_8\current_and_all_previous\";
factors = {"discomfort", "headache", "visual_stress"};
%factors = {'discomfort'};
for i =1:numel(factors)
    factor = factors{i};
    results_dir = super_dir + factor;
    experiments = dir(results_dir);
    
    % meta
    num_participants = NaN(1,100);
    
    % pos stuff
    pos_num_clusters = NaN(1,100);
    pos_p_value = NaN(1,100);
    pos_sum_t_value = NaN(1,100);
    pos_electrodes = {};c
    pos_time = NaN(1,100);
    pos_peak_t = NaN(1,100);
    pos_effect_size_r = NaN(1,100);
    pos_effect_size_cohens = NaN(1,100);
    
    % neg stuff
    neg_num_clusters = [];
    neg_p_value = [];
    neg_sum_t_value = [];
    neg_electrodes = {};
    neg_time = [];
    neg_peak_t = [];
    
    for i=1:size(experiments, 1)
        experiment = experiments(i).name;
        if contains(experiment, factor)
            experiment_number = split(experiment, '_');
            if strcmp(factor, 'visual_stress')
                experiment_number = str2num(experiment_number{4});
            else
                experiment_number = str2num(experiment_number{3});
            end
                
            main_dir = results_dir + "\" + experiment;
            cd(main_dir)
            load('stat.mat')
            load('number_of_participants.mat')
            
            num_participants(experiment_number) = num_part;
    
            if isfield(stat, 'posclusters') && numel(stat.posclusters) > 0
                most_positive = stat.posclusters(1);
                num_clusters = size(stat.posclusters,2);
                p = most_positive.prob;
                sum_of_t = most_positive.clusterstat;
                df = stat.df;
    
                load('pos_peak_level_stats_c_1.mat')
                electrode = pos_all_stats.electrodes{1};
                time = pos_all_stats.time(1);
                t = pos_all_stats.t_value(1);
    
                pos_num_clusters(experiment_number) = num_clusters;
    
                %if p <= 0.05
                pos_p_value(experiment_number) = p;
                pos_sum_t_value(experiment_number) = sum_of_t;
                pos_electrodes{experiment_number} = electrode;
                pos_time(experiment_number) = time;
                pos_peak_t(experiment_number) = t;
                pos_effect_size_r(experiment_number) = round(sqrt((t*t)/((t*t)+df)),2);
                pos_effect_size_cohens(experiment_number) = round((2*t)/sqrt(df),2);
                %else
                %    pos_num_clusters(experiment_number) = nan;
                %    pos_p_value(experiment_number) = nan;
                %    pos_sum_t_value(experiment_number) = nan;
                %    pos_electrodes{experiment_number} = "NaN";
                %    pos_time(experiment_number) = nan;
                %    pos_peak_t(experiment_number) = nan;
                %end
            else
                pos_num_clusters(experiment_number) = nan;
                pos_p_value(experiment_number) = nan;
                pos_sum_t_value(experiment_number) = nan;
                pos_electrodes{experiment_number} = 'No Electrode';
                pos_time(experiment_number) = nan;
                pos_peak_t(experiment_number) = nan;
            end
        end
    end
    
    
    t = tiledlayout(4,1, 'TileSpacing','Compact');
    nexttile;
    yyaxis left
    title("Illustration of Cluster Mass and P-values through time")
    scatter(1:100, pos_p_value, 45, 'filled', 'LineWidth',3);
    xlabel('number of trials')
    ylabel('p-values')
    [~, idx] = min(pos_p_value);
    %set(gca, 'YScale', 'log')
    ylim([0,0.25])
    
    yline(0.05, '--g', 'LineWidth',2.5)

    yyaxis right
    scatter(1:100, pos_sum_t_value, 45, 'filled', 'LineWidth',3);
    ylabel('Cluster Mass')
    xticks(1:100);
    grid on;
    
    xlabel('number of trials')
    xline(idx, '--r', 'LineWidth',2.5);
    legend({'P-values', '5%', 'Cluster Mass', 'Most Significant Cluster'},'Location','southeast')


    max_idc = max(pos_effect_size_cohens);
    max_idc = max_idc + 0.5;

    nexttile;
    yyaxis left
    title("Illustration of Cohen's d and Pearson's r through time")
    scatter(1:100, pos_effect_size_r, 45, 'filled','LineWidth',3);
    ylim([0,max_idc])
    xlabel('effect size calculations')
    ylabel('effect size r')
    xticks(1:100);
    yyaxis right
    scatter(1:100, pos_effect_size_cohens, 45, 'filled','LineWidth',3);
    ylabel('effect size cohens D')
    xline(idx, '--r', 'LineWidth',2.5);
    legend({'Effect size r', 'Cohens D', 'Most Significant Cluster'},'Location','southeast')
    ylim([0,max_idc])
    grid on;
    
    nexttile;
    scatter(1:100, pos_peak_t, 45, 'filled', 'LineWidth',3);
    
    xticks(1:100);
    xline(idx, '--r');
    title("Illustration of the largest T-value through time")
    legend({'Peak T-Value'},'Location','northwest')
    ylabel('Peak t-value');
    ylim([0, max(pos_peak_t)+.5])
   
    xlabel('number of trials')
     xline(idx, '--r', 'LineWidth',2.5);
    legend({'Peak T', 'Most Significant Cluster'},'Location','southeast')
    grid on;
    
    nexttile;
    unique_elecs = unique(pos_electrodes);
    n = size(pos_electrodes, 2);
    locaiton_vector = [];
    for i = 1:n
        elec = pos_electrodes{i};
        if strcmp(elec, nan)
            location_vector(i) = idx;
        else
            idx = find(contains(unique_elecs, elec));
            location_vector(i) = idx;
        end
    end

    scatter(1:100, location_vector, 45, 'filled', 'LineWidth',3);
    yticks([1:size(unique_elecs,2)])
    ylim([1,max(location_vector)]);
    yticklabels(unique_elecs);
    xticks(1:100);
    title("Illustration of Cluster Location through Time")
    xlabel('number of trials')
    ylabel('Electrode')
    grid on;

    %scatter(1:100, num_participants, 45, 'filled', 'LineWidth',3);
    %legend({'# Participants'},'Location','northwest')
    
    %xline(idx, '--r');
    %
    %legend({'Number of Participants'},'Location','northwest')
    %
    %xlabel('number of trials')
    %
    
    set(gcf,'Position',[100 100 2500 1250])
    save_dir = super_dir + factor + ".png";
    exportgraphics(gcf,save_dir,'Resolution',500);
    close;
end