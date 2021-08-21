%% set the paths and some extra info
clear all
clc
master_dir = 'C:\ProgramFiles\PhD\fieldtrip';
main_path = 'C:\ProgramFiles\PhD\participant_';
cd(master_dir);
rmpath C:\ProgramFiles\spm8;
addpath C:\ProgramFiles\spm12;
n_participants = 40;
partition.is_partition = 0;
partition.partition_number = 0;

%% decompose the data with cfg.output = 'fourier'
% converts to FT. Creates the wavelet and applies decomposition
% beware of memory, initally ran participant by participant

data_file = 'b1f1';
for  participant = [1]
    from_fieldtrip_to_spm(participant,main_path,data_file,partition);
end
% 
% %% load the data into 3 cell matrices
% for participant = 3
%     if ismember(participant, [6,13])
%            continue; 
%     end
%     participant_main_path = strcat(main_path, int2str(participant));
%     if exist(participant_main_path, 'dir')
%         cd(participant_main_path); 
%         load fourier_med
%         data_med{1} = TFRwave_med;
%         clear TFRwave_med
%         load fourier_thin
%         data_thin{1} = TFRwave_thin;
%         clear TFRwave_thin
%         load fourier_thick
%         data_thick{1} = TFRwave_thick;
%         clear TFRwave_thick
%     

        %% ITC analysis
%         data = [data_thin, data_med, data_thick];
%         clear data_thin data_med data_thick
%         for i = 1:3
%             itc = [];
%             itc.label = data{1,i}.label;
%             itc.freq = data{1,i}.freq;
%             itc.time = data{1,i}.time;
%             itc.dimord = 'chan_freq_time';
% 
%             F = data{1,i}.fourierspctrm;   % copy the Fourier spectrum
%             N = size(F,1);           % number of trials
% 
%             % compute inter-trial phase coherence (itpc)
%             itc.itpc = F./abs(F);         % divide by amplitude
%             itc.itpc = sum(itc.itpc,1);   % sum angles
%             itc.itpc = abs(itc.itpc)/N;   % take the absolute value and normalize
%             itc.itpc = squeeze(itc.itpc); % remove the first singleton dimension
% 
%             % compute inter-trial linear coherence (itlc)
%             itc.itlc = sum(F) ./ (sqrt(N*sum(abs(F).^2)));
%             itc.itlc = abs(itc.itlc);     % take the absolute value, i.e. ignore phase
%             itc.itlc = squeeze(itc.itlc); % remove the first singleton dimension
% 
%             itcs{participant,i} = itc;
% 
%         end
        %clear data;
%     end
% end

%% plot itpc values for each participant separately
% cd("C:\Users\marga\Desktop\Research Project\participants_Austyn")
% load itcs
% cd(master_dir);
% 
% for j = 1:3
%     for i = 1:32
%         itc = itcs{i,j};
%         fig=figure(j);
%         subplot(4,8,i)
%         imagesc(itc.time, itc.freq, squeeze(itc.itpc(1,:,:)));
%         axis xy
%         zlim([0 1]);
%         han=axes(fig,'visible','off'); 
%         han.Title.Visible='on';
%         han.XLabel.Visible='on';
%         han.YLabel.Visible='on';
%         ylabel(han,'Frequency (Hz)'); 
%         xlabel(han,'Time (s)');
%         title(han,'Inter-trial phase coherence for 32 participants');
%     end
% end

%% grand average of ITC values across participants
% cd("C:\Users\marga\Desktop\Research Project\participants_Austyn")
% load itcs
% cd(master_dir);
% 
% cfg = [];
% cfg.parameter = 'itpc';
% grand_avg_thin = ft_freqgrandaverage(cfg, itcs{:,1});
% grand_avg_thin.elec = data{1}.elec;
% grand_avg_med = ft_freqgrandaverage(cfg, itcs{:,2});
% grand_avg_med.elec = data{1}.elec;
% grand_avg_thick = ft_freqgrandaverage(cfg, itcs{:,3});
% grand_avg_thick.elec = data{1}.elec;
% 
% cfg = [];
% cfg.layout = grand_avg_thin.elec;
% cfg.colorbar = 'yes';
% cfg.colormap = 'jet';
% cfg.showlabels = 'yes';
% cfg.parameter = 'itpc';
% %cfg.zlim = [0.06 0.650]; %for multiplot
% cfg.zlim = [0.110 0.150]; %for topoplot
% figure(1)
% %ft_multiplotTFR(cfg, grand_avg_thin);
% ft_topoplotTFR(cfg, grand_avg_thin);
% figure(2)
% %ft_multiplotTFR(cfg, grand_avg_med);
% ft_topoplotTFR(cfg, grand_avg_med);
% figure(3)
% %ft_multiplotTFR(cfg, grand_avg_thick);
% ft_topoplotTFR(cfg, grand_avg_thick);

%% load the data for the 3 conditions (wavelet decomposition, outuput = power)
cd("C:\Users\marga\Desktop\Research Project\participants_Austyn")
load data_conditions
cd(master_dir)

for i = 1:32
    thin_data{i} = data_conditions{i}(1);
    med_data{i} = data_conditions{i}(2);
    thick_data{i} = data_conditions{i}(3);
end

%% calculate grand average for each frequency band, for each condition
freq_bins = {[8 13],[20 35],[30 45],[45 60],[60 80]};
grand_avgs = cell(3,5);
for f = 1:5
    cfg = [];
    cfg.channel   = 'all';
    cfg.foilim = freq_bins{f};
    cfg.toilim = [0 0.3];
    %cfg.parameter = 'itpc';
    cfg.parameter = 'powspctrm';
    %grand_avg1 = ft_freqgrandaverage(cfg, itcs{:,1});
    grand_avg1 = ft_freqgrandaverage(cfg, thin_data{:});
    grand_avg1.elec = data{1}.elec;
    %grand_avg2 = ft_freqgrandaverage(cfg, itcs{:,2});
    grand_avg2 = ft_freqgrandaverage(cfg, med_data{:});
    grand_avg2.elec = data{1}.elec;
    %grand_avg3 = ft_freqgrandaverage(cfg, itcs{:,3});
    grand_avg3 = ft_freqgrandaverage(cfg, thick_data{:});
    grand_avg3.elec = data{1}.elec;
    grand_avgs{1,f} = grand_avg1;
    grand_avgs{2,f} = grand_avg2;
    grand_avgs{3,f} = grand_avg3;
end

%% plot time series of power for conditions and calculate difference
thin_alpha = squeeze(mean(grand_avgs{1,1}.powspctrm,2,'omitnan'));
thin_beta = squeeze(mean(grand_avgs{1,2}.powspctrm,2,'omitnan'));
thin_lgamma = squeeze(mean(grand_avgs{1,3}.powspctrm,2,'omitnan'));
thin_mgamma = squeeze(mean(grand_avgs{1,4}.powspctrm,2,'omitnan'));
thin_hgamma = squeeze(mean(grand_avgs{1,5}.powspctrm,2,'omitnan'));

med_alpha = squeeze(mean(grand_avgs{2,1}.powspctrm,2,'omitnan'));
med_beta = squeeze(mean(grand_avgs{2,2}.powspctrm,2,'omitnan'));
med_lgamma = squeeze(mean(grand_avgs{2,3}.powspctrm,2,'omitnan'));
med_mgamma = squeeze(mean(grand_avgs{2,4}.powspctrm,2,'omitnan'));
med_hgamma = squeeze(mean(grand_avgs{2,5}.powspctrm,2,'omitnan'));

thick_alpha = squeeze(mean(grand_avgs{3,1}.powspctrm,2,'omitnan'));
thick_beta = squeeze(mean(grand_avgs{3,2}.powspctrm,2,'omitnan'));
thick_lgamma = squeeze(mean(grand_avgs{3,3}.powspctrm,2,'omitnan'));
thick_mgamma = squeeze(mean(grand_avgs{3,4}.powspctrm,2,'omitnan'));
thick_hgamma = squeeze(mean(grand_avgs{3,5}.powspctrm,2,'omitnan'));

figure(1)
title("Conditions' time series of power plotted on Oz, Alpha frequency band")
hold on
plot(grand_avgs{1}.time,thin_alpha(19,:),'Color','#0072BD','LineWidth',0.8)
plot(grand_avgs{1}.time,med_alpha(19,:),'Color','#D95319','LineWidth',0.8)
plot(grand_avgs{1}.time,thick_alpha(19,:),'Color','#EDB120','LineWidth',0.8)
xlim([0 0.3]); ylim([-3 4]); grid on
legend('Thin','Medium','Thick'); xlabel('Time (s)'); ylabel('Power (dB)');
hold off
figure(2)
title("Conditions' time series of power plotted on Oz, Beta frequency band")
hold on
plot(grand_avgs{1}.time,thin_beta(19,:),'Color','#0072BD','LineWidth',0.8)
plot(grand_avgs{1}.time,med_beta(19,:),'Color','#D95319','LineWidth',0.8)
plot(grand_avgs{1}.time,thick_beta(19,:),'Color','#EDB120','LineWidth',0.8)
xlim([0 0.3]); ylim([-3 4]); grid on
legend('Thin','Medium','Thick'); xlabel('Time (s)'); ylabel('Power (dB)');
hold off
figure(3)
title("Conditions' time series of power plotted on Oz, Low gamma frequency band")
hold on
plot(grand_avgs{1}.time,thin_lgamma(19,:),'Color','#0072BD','LineWidth',0.8)
plot(grand_avgs{1}.time,med_lgamma(19,:),'Color','#D95319','LineWidth',0.8)
plot(grand_avgs{1}.time,thick_lgamma(19,:),'Color','#EDB120','LineWidth',0.8)
xlim([0 0.3]); ylim([-3 4]); grid on
legend('Thin','Medium','Thick'); xlabel('Time (s)'); ylabel('Power (dB)');
hold off
figure(4)
title("Conditions' time series of power plotted on Oz, Medium gamma frequency band")
hold on
plot(grand_avgs{1}.time,thin_mgamma(19,:),'Color','#0072BD','LineWidth',0.8)
plot(grand_avgs{1}.time,med_mgamma(19,:),'Color','#D95319','LineWidth',0.8)
plot(grand_avgs{1}.time,thick_mgamma(19,:),'Color','#EDB120','LineWidth',0.8)
xlim([0 0.3]); ylim([-3 4]); grid on
legend('Thin','Medium','Thick'); xlabel('Time (s)'); ylabel('Power (dB)');
hold off
figure(5)
title("Conditions' time series of power plotted on Oz, High gamma frequency band")
hold on
plot(grand_avgs{1}.time,thin_hgamma(19,:),'Color','#0072BD','LineWidth',0.8)
plot(grand_avgs{1}.time,med_hgamma(19,:),'Color','#D95319','LineWidth',0.8)
plot(grand_avgs{1}.time,thick_hgamma(19,:),'Color','#EDB120','LineWidth',0.8)
xlim([0 0.3]); ylim([-3 4]); grid on
legend('Thin','Medium','Thick'); xlabel('Time (s)'); ylabel('Power (dB)');
hold off

%% differences between conditions
med_thin_diff_a = med_alpha-thin_alpha;
med_thin_diff_b = med_beta-thin_beta;
med_thin_diff_lg = med_lgamma-thin_lgamma;
med_thin_diff_mg = med_mgamma-thin_mgamma;
med_thin_diff_hg = med_hgamma-thin_hgamma;

med_thick_diff_a = med_alpha-thick_alpha;
med_thick_diff_b = med_beta-thick_beta;
med_thick_diff_lg = med_lgamma-thick_lgamma;
med_thick_diff_mg = med_mgamma-thick_mgamma;
med_thick_diff_hg = med_hgamma-thick_hgamma;

thin_thick_diff_a = thin_alpha-thick_alpha;
thin_thick_diff_b = thin_beta-thick_beta;
thin_thick_diff_lg = thin_lgamma-thick_lgamma;
thin_thick_diff_mg = thin_mgamma-thick_mgamma;
thin_thick_diff_hg = thin_hgamma-thick_hgamma;

figure(1)
title("Difference of power between conditions plotted on Oz, Alpha band")
hold on
plot(grand_avgs{1}.time,med_thin_diff_a(19,:),'Color','r','LineWidth',0.8)
plot(grand_avgs{1}.time,med_thick_diff_a(19,:),'Color','g','LineWidth',0.8)
plot(grand_avgs{1}.time,thin_thick_diff_a(19,:),'Color','b','LineWidth',0.8)
xlim([0 0.3]); ylim([-1.5 1.5]); grid on
legend('Medium-Thin','Medium-Thick','Thin-Thick'); xlabel('Time (s)'); ylabel('Power (dB)');
hold off
figure(2)
title("Difference of power between conditions plotted on Oz, Beta band")
hold on
plot(grand_avgs{1}.time,med_thin_diff_b(19,:),'Color','r','LineWidth',0.8)
plot(grand_avgs{1}.time,med_thick_diff_b(19,:),'Color','g','LineWidth',0.8)
plot(grand_avgs{1}.time,thin_thick_diff_b(19,:),'Color','b','LineWidth',0.8)
xlim([0 0.3]); ylim([-1.5 1.5]); grid on
legend('Medium-Thin','Medium-Thick','Thin-Thick'); xlabel('Time (s)'); ylabel('Power (dB)');
hold off
figure(3)
title("Difference of power between conditions plotted on Oz, Low gamma band")
hold on
plot(grand_avgs{1}.time,med_thin_diff_lg(19,:),'Color','r','LineWidth',0.8)
plot(grand_avgs{1}.time,med_thick_diff_lg(19,:),'Color','g','LineWidth',0.8)
plot(grand_avgs{1}.time,thin_thick_diff_lg(19,:),'Color','b','LineWidth',0.8)
xlim([0 0.3]); ylim([-1.5 1.5]); grid on
legend('Medium-Thin','Medium-Thick','Thin-Thick'); xlabel('Time (s)'); ylabel('Power (dB)');
hold off
figure(4)
title("Difference of power between conditions plotted on Oz, Medium gamma band")
hold on
plot(grand_avgs{1}.time,med_thin_diff_mg(19,:),'Color','r','LineWidth',0.8)
plot(grand_avgs{1}.time,med_thick_diff_mg(19,:),'Color','g','LineWidth',0.8)
plot(grand_avgs{1}.time,thin_thick_diff_mg(19,:),'Color','b','LineWidth',0.8)
xlim([0 0.3]); ylim([-1.5 1.5]); grid on
legend('Medium-Thin','Medium-Thick','Thin-Thick'); xlabel('Time (s)'); ylabel('Power (dB)');
hold off
figure(5)
title("Difference of power between conditions plotted on Oz, High gamma band")
hold on
plot(grand_avgs{1}.time,med_thin_diff_hg(19,:),'Color','r','LineWidth',0.8)
plot(grand_avgs{1}.time,med_thick_diff_hg(19,:),'Color','g','LineWidth',0.8)
plot(grand_avgs{1}.time,thin_thick_diff_hg(19,:),'Color','b','LineWidth',0.8)
xlim([0 0.3]); ylim([-1.5 1.5]); grid on
legend('Medium-Thin','Medium-Thick','Thin-Thick'); xlabel('Time (s)'); ylabel('Power (dB)');
hold off

%% plot grand average for each frequency band
lim = cell(1,5);
lim{1}=[0.125 0.235];
lim{2}=[0.110 0.150];
lim{3}=[0.110 0.140];
lim{4}=[0.110 0.135];
lim{5}=[0.110 0.130];
for f = 1:5
    cfg = [];
    cfg.layout = grand_avg_thin.elec;
    %cfg.colorbar = 'yes';
    cfg.colormap = 'jet';
    cfg.showlabels = 'yes';
    cfg.parameter = 'itpc';
    cfg.zlim = lim{f}; %for topoplot
    figure(f)
    subplot(1,3,1)
    ft_topoplotTFR(cfg, grand_avgs{1,f});
    subplot(1,3,2)
    ft_topoplotTFR(cfg, grand_avgs{2,f});
    subplot(1,3,3)
    ft_topoplotTFR(cfg, grand_avgs{3,f});   
end

%% plot scalp maps through time for itpc averaged across frequencies values
%% thin
lim = [0.09 0.52];
for f = 1:5
    figure(f)
    plot_scalp_maps(grand_avgs{1,f},lim);
end
%% medium
for f = 1:5
    figure(f)
    plot_scalp_maps(grand_avgs{2,f},lim); 
end
%% thick
for f = 1:5
    figure(f)
    plot_scalp_maps(grand_avgs{3,f},lim); 
end

%% plot scalp maps through time for itpc averaged across frequencies values
function plot_scalp_maps(grand_avg,lim)
    cfg = [];
    cfg.layout = grand_avg.elec;
    %cfg.colorbar = 'yes';
    cfg.colormap = 'jet';
    cfg.showlabels = 'yes';
    cfg.parameter = 'itpc';
    cfg.xlim = 0:0.025:0.3;
    cfg.zlim = lim;
    ft_topoplotTFR(cfg, grand_avg);
end

%% return the SPM data in FieldTrip format
function from_fieldtrip_to_spm(pax, main_path, filename, partition)

    for participant = pax
        
        
        participant_main_path = strcat(main_path, int2str(participant));
        if exist(participant_main_path, 'dir')
            if participant < 10
               p = strcat('0', int2str(participant));
            else
               p = int2str(participant);
            end

            % data structure
            data_structure = strcat('spmeeg_P', p);  
            data_structure = strcat(data_structure, '_075_80hz_rejected_tempesta.mat');
            data_structure = strcat(filename, data_structure); 
            cd(participant_main_path); 
            cd('SPM_ARCHIVE');
            
            load('b1f1spmeeg_P01_075_80Hz.mat');
    

            spm_eeg = meeg(D);
            fieldtrip_raw = spm_eeg.ftraw;
            
            n_trials = size(D.trials);
            n_trials = n_trials(2);
            mt=1; tht=1; tt=1;
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
                        med(mt) = fieldtrip_raw.trial(index_i);
                        mt=mt+1;
                    elseif contains(label, 'thin')
                        thin(tht) = fieldtrip_raw.trial(index_i);
                        tht=tht+1;
                    elseif contains(label, 'thick')
                        thick(tt) = fieldtrip_raw.trial(index_i);
                        tt=tt+1;
                     end
                end
            end
            
            % update the fieldtrip structure with fields of information
            raw_med = [];
            raw_med.label = fieldtrip_raw.label;
            raw_med.elec = fieldtrip_raw.elec;
            raw_med.trial = med;
            raw_med.time = fieldtrip_raw.time(1:length(raw_med.trial));
            raw_med.dimord = 'chan_time';
            raw_med = remove_electrodes(raw_med);
            
            raw_thin = [];
            raw_thin.label = fieldtrip_raw.label;
            raw_thin.elec = fieldtrip_raw.elec;
            raw_thin.trial = thin;
            raw_thin.time = fieldtrip_raw.time(1:length(raw_thin.trial));
            raw_thin.dimord = 'chan_time';
            raw_thin = remove_electrodes(raw_thin);
            
            raw_thick = [];
            raw_thick.label = fieldtrip_raw.label;
            raw_thick.elec = fieldtrip_raw.elec;
            raw_thick.trial = thick;
            raw_thick.time = fieldtrip_raw.time(1:length(raw_thick.trial));
            raw_thick.dimord = 'chan_time';
            raw_thick = remove_electrodes(raw_thick);
            
            %wavelet decomposition
            cfg = [];
            cfg.channel = 'eeg';
            cfg.method = 'wavelet';
            cfg.width = 5;
            cfg.output = 'fourier';
            cfg.pad = 'nextpow2';
            cfg.foi = 5:60;
            cfg.toi = -0.2:0.002:1.2;
            cfg.keeptrials = 'yes';
            TFRwave_med = ft_freqanalysis(cfg, raw_med);
            TFRwave_med.info = 'medium'; 
            %crop the epoch
            TFRwave_med.time = TFRwave_med.time(200:651);
            TFRwave_med.fourierspctrm = TFRwave_med.fourierspctrm(:,:,:,200:651);
            save('fourier_med.mat', 'TFRwave_med', '-v7.3')
            clear TFRwave_med
            TFRwave_thin = ft_freqanalysis(cfg, raw_thin);
            TFRwave_thin.info = 'thin'; 
            %crop
            TFRwave_thin.time = TFRwave_thin.time(200:651);
            TFRwave_thin.fourierspctrm = TFRwave_thin.fourierspctrm(:,:,:,200:651);
            save('fourier_thin.mat', 'TFRwave_thin', '-v7.3')
            clear TFRwave_thin
            TFRwave_thick = ft_freqanalysis(cfg, raw_thick);
            TFRwave_thick.info = 'thick'; 
            %crop
            TFRwave_thick.time = TFRwave_thick.time(200:651);
            TFRwave_thick.fourierspctrm = TFRwave_thick.fourierspctrm(:,:,:,200:651);
            save('fourier_thick.mat', 'TFRwave_thick', '-v7.3')
            clear TFRwave_thick
            
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
            for i=1:length(fieldtrip_raw.trial)
                new_trial{1,i}(cnt, :) = fieldtrip_raw.trial{1,i}(idx,:);
            end
            cnt = cnt+1;
        end
    end
    
    fieldtrip_raw.label = new_elec;
    fieldtrip_raw.trial = new_trial;
end