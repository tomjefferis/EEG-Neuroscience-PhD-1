super_dir = "D:\PhD\results\trial_level_2_8\current_and_all_previous\";
factors = {"discomfort", "headache", "visual_stress"};
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
    pos_electrodes = {};
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
    
            if isfield(stat, 'negclusters') && numel(stat.negclusters) > 0
                most_positive = stat.negclusters(1);
                num_clusters = size(stat.negclusters,2);
                p = most_positive.prob;
                sum_of_t = most_positive.clusterstat;
                df = stat.df;
    
                load('neg_peak_level_stats.mat')
                electrode = neg_all_stats.electrodes{1};
                time = neg_all_stats.time(1);
                t = neg_all_statss.t_value(1);
    
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
                pos_electrodes{experiment_number} = "NaN";
                pos_time(experiment_number) = nan;
                pos_peak_t(experiment_number) = nan;
            end
        end
    end
    
    
    t = tiledlayout(4,1, 'TileSpacing','Compact');
    nexttile;
    yyaxis left
    scatter(1:100, pos_p_value, 45, 'filled', 'LineWidth',3);
    xlabel('number of trials')
    ylabel('p-values')
    [~, idx] = min(pos_p_value);
    %set(gca, 'YScale', 'log')
    ylim([0,1])
    yline(0.05, '--r', 'LineWidth',2.5)
    yyaxis right
    scatter(1:100, pos_sum_t_value, 45, 'filled', 'LineWidth',3);
    ylabel('sum of t values')
    xticks(1:100);
    grid on;
    xline(idx, '--r', 'LineWidth',2.5);
    xlabel('number of trials')
    legend({'P-values', 'Sum of T-values', 'Peak Trial Count'},'Location','northeast')
    
    max_idc = max(pos_effect_size_cohens);
    max_idc = max_idc + 0.5;

    nexttile;
    yyaxis left
    scatter(1:100, pos_effect_size_r, 45, 'filled','LineWidth',3);
    ylim([0,max_idc])
    xlabel('effect size calculations')
    ylabel('effect size r')
    xticks(1:100);
    yyaxis right
    scatter(1:100, pos_effect_size_cohens, 45, 'filled','LineWidth',3);
    ylabel('effect size cohens D')
    ylim([0,max_idc])
    grid on;
    
    nexttile;
    scatter(1:100, pos_peak_t, 45, 'filled', 'LineWidth',3);
    
    xticks(1:100);
    xline(idx, '--r');
    legend({'Peak T-Value'},'Location','northwest')
    ylabel('Peak t-value');
    ylim([0, 4])
    xlabel('number of trials')
    grid on;
    
    nexttile;
    scatter(1:100, num_participants, 45, 'filled', 'LineWidth',3);
    legend({'# Participants'},'Location','northwest')
    xticks(1:100);
    xline(idx, '--r');
    legend({'Number of Participants'},'Location','northwest')
    ylabel('Count of participants')
    xlabel('number of trials')
    grid on;
    
    set(gcf,'Position',[100 100 2500 1250])
    save_dir = super_dir + factor + ".png";
    exportgraphics(gcf,save_dir,'Resolution',500);
    close;
end