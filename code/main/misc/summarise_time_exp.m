clear all;
master_dir =  "D:\PhD\results\time_domain";
experiment_types = {'mean_intercept', 'partitions', 'onsets', 'pure-factor-effect'};
factors = {'visual_stress', 'discomfort', 'headache', 'no-factor'};
frequency_bands = ["sensitization", "habituation"];


cluster_no = [];
p_values = [];
cluster_size_mtx = [];
t_values = [];
peak_electrodes = {};
times = [];
factor_typess = {};

cnt = 1;
for exp = 1:numel(experiment_types)
    experiment = experiment_types{exp};
    if strcmp(experiment, 'mean_intercept')
        for band = 1:numel(frequency_bands)
            fb = frequency_bands(band);

            path_to_experiment = master_dir + "\" + experiment + "\" + "no-factor_" + fb;
            cd(path_to_experiment)
            load("stat.mat")

            pos_stat = stat.posclusters;
            pos_stat = pos_stat([pos_stat.prob] <= 0.25);
            
            if numel(pos_stat) > 0
                num_clusters = size(pos_stat, 2);
                for nc = 1:num_clusters
                    p_value = stat.posclusters(nc).prob;
                    cluster_size = stat.posclusters(nc).clusterstat;
                    first_cluster = "pos_peak_level_stats_c_" + nc + ".mat";
                    load(first_cluster)
                    peak_electrode = pos_all_stats.electrodes(nc);
                    time = pos_all_stats.time(nc);
                    t = pos_all_stats.t_value(nc);

                    cluster_no(cnt) = nc;
                    p_values(cnt) = p_value;
                    cluster_size_mtx(cnt) = cluster_size;
                    t_values(cnt) = t;
                    peak_electrodes{cnt} = peak_electrode{1};
                    times(cnt) = time;
                    factor_typess{cnt} = experiment + "_" + fb;
                       cnt = cnt + 1;
                end
            end
        end
    elseif strcmp(experiment, 'partitions') || strcmp(experiment, 'onsets')
        for f = 1:numel(factors)
            fact = factors(f);
            for band = 1:numel(frequency_bands)
                fb = frequency_bands(band);
                path_to_experiment = master_dir + "\" + experiment + "\" + fact + "_" + fb;
                cd(path_to_experiment)
                load("stat.mat")
    
                pos_stat = stat.posclusters;
                pos_stat = pos_stat([pos_stat.prob] <= 0.25);
                
                if numel(pos_stat) > 0
                    num_clusters = size(pos_stat, 2);
                    for nc = 1:num_clusters
                        p_value = stat.posclusters(nc).prob;
                        cluster_size = stat.posclusters(nc).clusterstat;
                        first_cluster = "pos_peak_level_stats_c_" + nc + ".mat";
                        load(first_cluster)
                        peak_electrode = pos_all_stats.electrodes(nc);
                        time = pos_all_stats.time(nc);
                        t = pos_all_stats.t_value(nc);
    
                        cluster_no(cnt) = nc;
                        p_values(cnt) = p_value;
                        cluster_size_mtx(cnt) = cluster_size;
                        t_values(cnt) = t;
                        peak_electrodes{cnt} = peak_electrode{1};
                        times(cnt) = time;
                        factor_typess{cnt} = experiment + "_" + fact + "_" + fb;

                        cnt = cnt + 1;

                    end
                end
            end
        end
    elseif strcmp(experiment, 'factor-effect')
        for f = 1:numel(factors)
            fact = factors(f);
            for band = 1:numel(frequency_bands)
                if ~strcmp(fact, 'no-factor')
                    fb = frequency_bands(band);
                    path_to_experiment = master_dir + "\" + experiment + "\" + fact + "_p1_" + fb;
                    cd(path_to_experiment)
                    load("stat.mat")
                    pos_stat = stat.posclusters;
                    pos_stat = pos_stat([pos_stat.prob] <= 0.25);
                    
                    if numel(pos_stat) > 0
                        num_clusters = size(pos_stat, 2);
                        for nc = 1:num_clusters
                            p_value = stat.posclusters(nc).prob;
                            cluster_size = stat.posclusters(nc).clusterstat;
                            first_cluster = "pos_peak_level_stats_c_" + nc + ".mat";
                            load(first_cluster)
                            peak_electrode = pos_all_stats.electrodes(nc);
                            time = pos_all_stats.time(nc);
                            t = pos_all_stats.t_value(nc);
        
                            cluster_no(cnt) = nc;
                            p_values(cnt) = p_value;
                            cluster_size_mtx(cnt) = cluster_size;
                            t_values(cnt) = t;
                            peak_electrodes{cnt} = peak_electrode{1};
                            times(cnt) = time;
                            factor_typess{cnt} = experiment + "_" + fact + "_" + fb;
    
                            cnt = cnt + 1;
    
                        end
                    end
                end
            end
        end
    end

end

summary = table(cluster_no', p_values', cluster_size_mtx', t_values', peak_electrodes', times', factor_typess');