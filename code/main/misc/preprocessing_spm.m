%% preprocessing pipeline for SPM *** Author; Cihan Dogan
clear all
cd("D:\PhD");
addpath('C:\External_Software\fieldtrip-20210807');
addpath('C:\External_Software\spm12')

%% Change these variables depending on what you would like to do.
%% preprocessing pipeline for FieldTrip *** Author; Cihan Dogan
clear all;
restoredefaultpath;
addpath('C:\External_Software\fieldtrip-20210807');
addpath('C:\External_Software\spm12')
ft_defaults;
cd("D:\PhD");
    
%% Change these variables depending on what you would like to do.
main_path = 'D:\PhD\participant_';
to_preprocess = {'partitions'};
type_of_analysis = 'frequency_domain'; % or time_domain

onsets = [
    2,3,4,5,6,7,8
];
number_of_onsets = size(onsets);
number_of_onsets = number_of_onsets(1);


%% main preprocessing loop
for k=1:numel(to_preprocess)
    
    analysis_type=to_preprocess{k};
    
    if strcmp(analysis_type, 'mean_intercept') || strcmp(analysis_type, 'eye_confound')
        n_participants = 40;
    elseif contains(analysis_type, 'partitions')
        n_participants = 40;
    end
    
    % parfloor

    [n_onsets, ~] = size(onsets);
    for i=1:n_onsets
        subset_onsets = onsets(i,:);
        for participant = 13:n_participants

            %% gets the onsets of interest
            [thin, med, thick, description] = get_onsets(subset_onsets, analysis_type);
            full_description = strcat(analysis_type, '_', description{1});
            full_description = strcat(type_of_analysis, {'_'}, full_description);
            full_description = full_description{1};

            %% works out where to load the data
            participant_main_path = strcat(main_path, int2str(participant));    

            if exist(participant_main_path, 'dir')
                participant_main_path = strcat(participant_main_path, '\');
                data_structure = 'spmeeg_P';        

                if participant < 10
                    p = strcat('0', int2str(participant));
                else
                   p = int2str(participant);
                end

                data_structure = strcat(data_structure, p);
                data_structure = strcat(data_structure, '_075_80Hz_rejected.mat');   
                file_main_path = strcat(participant_main_path, data_structure);
                save_path = strcat(participant_main_path, 'SPM_ARCHIVE\');
                cd(participant_main_path);

                if ~isfile(data_structure)
                    continue;
                end

                %% this function updates the trial information so that you only
                % analyse the conditions of interest
                condition_names = label_data(thin, med, thick, participant_main_path,  data_structure, analysis_type);

                %% sort out paths
                load(file_main_path);
                D.data.fname = convertStringsToChars(strrep(file_main_path, "mat", "dat"));
                save(file_main_path, 'D')


                %% filter
                matlabbatch{1}.spm.meeg.preproc.filter.D = {file_main_path};
                matlabbatch{1}.spm.meeg.preproc.filter.type = 'butterworth';
                matlabbatch{1}.spm.meeg.preproc.filter.band = 'bandpass';
                matlabbatch{1}.spm.meeg.preproc.filter.freq = [0.5 80];
                matlabbatch{1}.spm.meeg.preproc.filter.dir = 'twopass';
                matlabbatch{1}.spm.meeg.preproc.filter.order = 5;
                matlabbatch{1}.spm.meeg.preproc.filter.prefix = strcat(save_path, 'f1');
                spm_jobman('run',matlabbatch)
                clear matlabbatch

                %% baseline
                %data_structure = strcat('f1', data_structure);
                %file_main_path = strcat(save_path, data_structure);
                %matlabbatch{1}.spm.meeg.preproc.bc.D = {file_main_path};
                %matlabbatch{1}.spm.meeg.preproc.bc.timewin = [-200 0];
                %matlabbatch{1}.spm.meeg.preproc.bc.prefix = 'b1';
                %spm_jobman('run',matlabbatch)
                %clear matlabbatch

                %% threshold
                data_structure = strcat('f1', data_structure);
                file_main_path = strcat(save_path, data_structure);
                matlabbatch{1}.spm.meeg.preproc.artefact.D = {file_main_path};
                matlabbatch{1}.spm.meeg.preproc.artefact.mode = 'reject';
                matlabbatch{1}.spm.meeg.preproc.artefact.badchanthresh = 0.5;
                matlabbatch{1}.spm.meeg.preproc.artefact.append = true;
                matlabbatch{1}.spm.meeg.preproc.artefact.methods.channels{1}.type = 'EEG';
                matlabbatch{1}.spm.meeg.preproc.artefact.methods.fun.threshchan.threshold = 100;
                matlabbatch{1}.spm.meeg.preproc.artefact.methods.fun.threshchan.excwin = 1000;
                matlabbatch{1}.spm.meeg.preproc.artefact.prefix = strcat(save_path, 't1');
                spm_jobman('run',matlabbatch)
                clear matlabbatch


                %% determine whether we need to reject an entire particiapnt
                data_structure = strcat('t1', data_structure);
                dir = strcat(save_path, data_structure);
                reject_participant = reject_particiapnt_based_on_bad_trials(dir);
                 
                 if reject_participant == 1
                     disp('Rejected participant not enough trials after 100 uv rejection');
                     delete(dir);
                     delete(replace(dir, '.mat', '.dat'));
                     continue;
                 end

                %% load and convert from SPM > FieldTrip
                load(dir);
                spm_eeg = meeg(D);
                raw = spm_eeg.ftraw;


                %% label with the trial order (for analysis later)
                raw = label_data_with_trial_order(raw, D);

                %% label with the trial order (duplicate)
                raw = label_data_with_trials(raw, raw);


                % relabel the relevant conditions after postprocessing
                raw = relabel_conditions(raw, D);

                % get the data ready for FT analysis
                [trial_level, grand_averages] = data_ready_for_analysis(raw, analysis_type);
                
                % saves the grand average data (easier to load rather than
                % trial level. Also saves trial level if needed.
                participant_main_path = strcat(participant_main_path, "SPM_ARCHIVE\");
                save_data(grand_averages, participant_main_path, full_description, '_grand-average')
                save_data(trial_level, participant_main_path, full_description, '_trial-level')

                disp(strcat('PROCESSED PARTICIPANT..',int2str(participant))); 

            end
        end
    end
end


%% label the data with the trial order for analysis later
function raw = label_data_with_trial_order(raw, D)
    num_trials = size(raw.sampleinfo, 1);

    for onset =1:num_trials
        t_order = D.trials(onset).trial_order;
        raw.sampleinfo(onset, 3) = t_order;
    end
end

%% label and create the grand average dataset
function [trial_level, grand_averages] = data_ready_for_analysis(postprocessed, data_type)

    postprocessed = remove_electrodes(postprocessed, data_type);

    idx_used_for_saving_data = 1; 
    trial_names_and_order = postprocessed.trial_order;
    sample_information = postprocessed.sampleinfo;

    % find the condition labels used to match up the data
    if strcmp(data_type, 'partitions')
        p1_thin_idx = find(contains(trial_names_and_order,'_partition_1__thin'));
        p1_thick_idx = find(contains(trial_names_and_order,'_partition_1__thick'));
        p1_med_idx = find(contains(trial_names_and_order,'_partition_1__medium'));
        p2_thin_idx = find(contains(trial_names_and_order,'_partition_2__thin'));
        p2_thick_idx = find(contains(trial_names_and_order,'_partition_2__thick'));
        p2_med_idx = find(contains(trial_names_and_order,'_partition_2__medium'));
        p3_thin_idx = find(contains(trial_names_and_order,'_partition_3__thin'));
        p3_thick_idx = find(contains(trial_names_and_order,'_partition_3__thick'));
        p3_med_idx = find(contains(trial_names_and_order,'_partition_3__medium'));
        % put the trials into the respective buckets
        trials = postprocessed.trial;
        [~,n] = size(trials);

        [p1_thin, p1_thick, p1_med, p2_thin, p2_thick, p2_med, ...
            p3_thin, p3_thick, p3_med] = deal([], [], [], [], [], [], [], [], []);
        [p1_thin_order, p1_thick_order, p1_med_order, p2_thin_order, p2_thick_order, ...
            p2_med_order, p3_thin_order, p3_thick_order, p3_med_order] ...
            = deal([], [], [], [], [], [], [], [], []);

        for idx=1:n
            trial = trials{idx};
            condition = sample_information(idx, 3);
            trial_order = sample_information(idx, 4);
            if condition == p1_thin_idx
                p1_thin(:,:,end+1) = trial;
                p1_thin_order(:,:,end+1) = trial_order;
            elseif condition == p2_thin_idx
                p2_thin(:,:,end+1) = trial;
                p2_thin_order(:,:,end+1) = trial_order;
            elseif condition == p3_thin_idx
                p3_thin(:,:,end+1) = trial;
                p3_thin_order(:,:,end+1) = trial_order;
            elseif condition == p1_thick_idx
                p1_thick(:,:,end+1) = trial;
                p1_thick_order(:,:,end+1) = trial_order;
            elseif condition == p2_thick_idx
                p2_thick(:,:,end+1) = trial;
                p2_thick_order(:,:,end+1) = trial_order;
            elseif condition == p3_thick_idx
                p3_thick(:,:,end+1) = trial;
                p3_thick_order(:,:,end+1) = trial_order;
            elseif condition == p1_med_idx
                p1_med(:,:,end+1) = trial;
                p1_med_order(:,:,end+1) = trial_order;
            elseif condition == p2_med_idx
                p2_med(:,:,end+1) = trial;
                p2_med_order(:,:,end+1) = trial_order;
            elseif condition == p3_med_idx
                p3_med(:,:,end+1) = trial;
                p3_med_order(:,:,end+1) = trial_order;
            end
        end

        trial_level.p1_med = convert_to_fieldtrip_format(p1_med);
        trial_level.p2_med = convert_to_fieldtrip_format(p2_med);
        trial_level.p3_med = convert_to_fieldtrip_format(p3_med);
        trial_level.p1_thin = convert_to_fieldtrip_format(p1_thin);
        trial_level.p2_thin = convert_to_fieldtrip_format(p2_thin);
        trial_level.p3_thin = convert_to_fieldtrip_format(p3_thin);
        trial_level.p1_thick = convert_to_fieldtrip_format(p1_thick);
        trial_level.p2_thick = convert_to_fieldtrip_format(p2_thick);
        trial_level.p3_thick = convert_to_fieldtrip_format(p3_thick);

        trial_level.p1_med_order = p1_med_order;
        trial_level.p2_med_order = p2_med_order;
        trial_level.p3_med_order = p3_med_order;
        trial_level.p1_thin_order = p1_thin_order;
        trial_level.p2_thin_order = p2_thin_order;
        trial_level.p3_thin_order = p3_thin_order;
        trial_level.p1_thick_order = p1_thick_order;
        trial_level.p2_thick_order = p2_thick_order;
        trial_level.p3_thick_order = p3_thick_order;


        trial_level.elec = postprocessed.elec;
        trial_level.label = postprocessed.label;
        trial_level.time = postprocessed.time(1,1);

        % calculate the means
        p1_med = mean(p1_med, 3);
        p2_med = mean(p2_med, 3);
        p3_med = mean(p3_med, 3);
        p1_thin = mean(p1_thin, 3);
        p2_thin = mean(p2_thin, 3);
        p3_thin = mean(p3_thin, 3);
        p1_thick = mean(p1_thick, 3);
        p2_thick = mean(p2_thick, 3);
        p3_thick = mean(p3_thick, 3);

        % setup the data structure for analysis
        grand_averages.p1_pgi = p1_med - (p1_thin+p1_thick)/2;
        grand_averages.p2_pgi = p2_med - (p2_thin+p2_thick)/2;
        grand_averages.p3_pgi = p3_med - (p3_thin+p3_thick)/2;

        grand_averages.p1_med = p1_med;
        grand_averages.p2_med = p2_med;
        grand_averages.p3_med = p3_med;

        grand_averages.p1_thin = p1_thin;
        grand_averages.p2_thin = p2_thin;
        grand_averages.p3_thin = p3_thin;

        grand_averages.p1_thick = p1_thick;
        grand_averages.p2_thick = p2_thick;
        grand_averages.p3_thick = p3_thick;

        grand_averages.trialinfo = [1];
        grand_averages.time = postprocessed.time(1,1);
        grand_averages.elec = postprocessed.elec;
        grand_averages.dimord = 'chan_time';
        grand_averages.label = postprocessed.label;
    elseif strcmp(data_type, 'mean_intercept') || strcmp(data_type, 'eye_confound')
        thin_idx = find(contains(trial_names_and_order,'thin'));
        med_idx = find(contains(trial_names_and_order,'medium'));
        thick_idx = find(contains(trial_names_and_order,'thick'));

        trials = postprocessed.trial;
        [~,n] = size(trials);
        
        [thin, medium, thick] = deal([],[],[]);

        [thin_order, medium_order, thick_order] = deal([],[],[]);
        
        for idx=1:n
            trial = trials{idx};
            condition = sample_information(idx, 3);
            trial_order = sample_information(idx, 4);
            if condition == thin_idx
                thin(:,:,end+1) = trial;  
                thin_order(:,:,end+1) = trial_order;
            elseif condition == med_idx
                medium(:,:,end+1) = trial;
                medium_order(:,:,end+1) = trial_order;
            elseif condition == thick_idx
                thick(:,:,end+1) = trial;
                thick_order(:,:,end+1) = trial_order;
            end
        end
    

        trial_level.thin = convert_to_fieldtrip_format(thin);
        trial_level.med = convert_to_fieldtrip_format(medium);
        trial_level.thick = convert_to_fieldtrip_format(thick);
        trial_level.thin_order = thin_order;
        trial_level.thick_order = thick_order;
        trial_level.med_order = medium_order;
        trial_level.elec = postprocessed.elec;
        trial_level.time = postprocessed.time(1,1);
        trial_level.label = postprocessed.label;

        % calculate means
        thin = mean(thin, 3);
        thick = mean(thick, 3);
        medium = mean(medium, 3);

        grand_averages.thin = thin;
        grand_averages.thick = thick;
        grand_averages.med = medium;
        grand_averages.trialinfo = [1];
        grand_averages.time = postprocessed.time(1,1);
        grand_averages.elec = postprocessed.elec;
        grand_averages.dimord = 'chan_time';
        grand_averages.label = postprocessed.label;
    end
end
%% remvoe electrodes
function postprocessed = remove_electrodes(postprocessed, type)
    
    if strcmp(type, 'eye_confound')
        to_remove = {
    'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14', ...
    'A15','A16','A17','A18','A19','A20','A21','A22','A23','A24','A25','A26','A27', ...
    'A28','A29','A30','A31','A32','B1','B2','B3','B4','B5','B6','B7','B8','B9', ...
    'B10','B11','B12','B13','B14','B15','B16','B17','B18','B19','B20','B21','B22', ...
    'B23','B24','B25','B26','B27','B28','B29','B30','B31','B32','C1','C2','C3', ...
    'C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18', ...
    'C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31', ...
    'C32','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13', ...
    'D14','D15','D16','D17','D18','D19','D20','D21','D22','D23','D24','D25','D26', ...
    'D27','D28','D29','D30','D31','D32', 'EXG5', 'EXG6'
    };
    else
        to_remove = {'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'HEOG', 'VEOG'};
    end



    electorode_information = postprocessed.elec;
    
    trials = postprocessed.trial;
    [~,n] = size(trials);
    
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
    
    postprocessed.elec.chanpos = new_chanpos;
    postprocessed.elec.chantype = new_chantype;
    postprocessed.elec.chanunit = new_chanunit;
    postprocessed.elec.elecpos = new_elecpos;
    postprocessed.elec.label = new_labels;
   
    new_trials = {};
    for i = 1:n
        t = trials{i}; 
        cnt = 1;
        for idx = 1:numel(postprocessed.label)
            elec = postprocessed.label{idx};
            if ~ismember(elec, to_remove)
                new_elec{cnt} = elec;
                new_t(cnt, :) = t(idx, :);
                cnt = cnt + 1;
            end
        end
        new_trials{i} = new_t;
    end
        
    postprocessed.label = new_elec;
    postprocessed.trial = new_trials;
end

%% save the grand average data
function save_data(data, participant_main_path, description, data_type)
    path = strcat(participant_main_path, description, data_type, '.mat');
    save(path, 'data');
end

%% rename the trials with the correct name
function postprocessed = relabel_conditions(postprocessed, D)
    [~, n_trials] = size(D.trials);
    
    label_names = {};
    
    cnt = 1;
    for i = 1:n_trials
        trial = D.trials(i).label;
        
        if ~any(strcmp(label_names,trial))
            label_names{cnt} = trial;
            cnt = cnt+1;
        end
    end
   
    postprocessed.trial_order = label_names;
end

%% update the samples with trial info
function postprocessed = label_data_with_trials(raw, postprocessed)
    original_info = raw.sampleinfo;
    original_labels = raw.sampleinfo(:,3);
    original_info(:,3) = raw.trialinfo';
    original_info(:,4) = original_labels;
    new_info = postprocessed.sampleinfo;
    
    [row, ~] = size(new_info);
    
    for i=1:row
       start_sample = new_info(i,1);
       idx = find(original_info(:,1)==start_sample);
       original_label = original_info(idx,3);
       original_trial_point_in_time = original_info(idx,4);
       new_info(i,3) = original_label;
       new_info(i,4) = original_trial_point_in_time;
    end
    
    postprocessed.sampleinfo = new_info;
end
%% get the eeg channels excluding EOG etc
function labels = get_eeg_channels(data, type)

    if contains(type, 'partitions') || strcmp(type, 'mean_intercept')
        to_remove = {'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'HEOG', 'VEOG'};
        to_remove = {'A11', 'A12', 'A13', 'A14', 'A24', 'A25', 'A26','A27', 'B8', 'B9','EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'HEOG', 'VEOG'};
    elseif strcmp(type, 'eye_confound')
        to_remove = {'A11', 'A12', 'A13', 'A14', 'A24', 'A25', 'A26','A27', 'B8', 'B9','EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'HEOG', 'VEOG'};
    end
    
    labels = setdiff(data.elec.label, to_remove);
end

%% return the desired onsets
function [thin, med, thick, description] = get_onsets(onsets, analysis_type)
    % below is purely a reference so we have all codes
    onsets_thin_REF = {'65411'; '65412'; '65413'; '65414'; '65415'; '65416';'65417'; '65418'; '65419'};
    onsets_medium_REF = {'65401'; '65402'; '65403'; '65404'; '65405'; '65406'; '65407'; '65408'; '65409'};
    onsets_thick_REF = {'65391'; '65392'; '65393'; '65394'; '65395'; '65396'; '65397'; '65398'; '65399'};
    
    thin = onsets_thin_REF(onsets);
    med = onsets_medium_REF(onsets);
    thick = onsets_thick_REF(onsets);
    
    shape = size(onsets);
    number_of_onsets = shape(2);

    description = 'onsets';
    for i = 1:number_of_onsets
       onset = int2str(onsets(i));
       description = strcat(description, {'_'}, onset);
    end
    
    if strcmp(analysis_type, 'partitions')
        str = '';
        cnt =1;
        for o=onsets
            str = strcat(str,int2str(o));
            str = strcat(str, '_');
        end
        str = str(1:end-1);
        description = {strcat('partitioned_onsets', {'_'}, str)}; 
        description = description{1};
    end
end

%% updates the EEG data with the onsets we are interested in analysing
function condition_names = label_data(thin, medium, thick, path, fname, analysis_type)
    factor_name = '';
    file = strcat(path, fname);
    load(file); % loads the D object
    D.fname = fname;
    D.path = path;
    n_trials = size(D.trials);
    n_trials = n_trials(2);
    count = 1;
    condition_names = {};
    
    if ~strcmp(analysis_type, 'partitions')
        med_order = 1;
        thick_order = 1;
        thin_order = 1;
        for onset = 1:n_trials                              
            
            events = D.trials(onset).events;
            [~, rows] = size(events);
            for i = 1:rows
                condition = events(i).binlabel;
                if ~strcmp(condition, '""')
                    condition_found = 1;
                    break
                end
            end

            if sum(contains(condition, thin))
                condition = strcat(factor_name, '_thin');
                D.trials(onset).trial_order = thin_order;  
                thin_order = thin_order + 1;
            elseif sum(contains(condition, medium))
                condition = strcat(factor_name, '_medium');
                D.trials(onset).trial_order = med_order;  
                med_order = med_order + 1;
            elseif sum(contains(condition, thick))
                condition = strcat(factor_name, '_thick');
                D.trials(onset).trial_order = thick_order;  
                thick_order = thick_order + 1;
            else
                condition = 'N/A';
                D.trials(onset).trial_order = -999; 
            end

            condition_names{onset} = condition;
            D.trials(onset).label = condition;    
            count = count + 1;
        end      
    else        
        partition_number = 1;
        max_epoch = D.trials(n_trials).events.epoch;
        med_order = 1;
        thick_order = 1;
        thin_order = 1;
        for onset = 1:n_trials
            events = D.trials(onset).events;
            current_epoch = D.trials(onset).events.epoch;
            
            [~, rows] = size(events);
            for i = 1:rows
                condition = events(i).binlabel;
                if ~strcmp(condition, '""')
                    condition_found = 1;
                    break
                end
            end
            
            if ~condition_found == 1
                error('Condition not found...');
            end
            
            if current_epoch <= (max_epoch/3)
                partition_number = 1;
            elseif (current_epoch >(max_epoch/3)) && (current_epoch <=(max_epoch/3)*2)
                partition_number = 2;
            elseif (current_epoch >(max_epoch/3)*2) && (current_epoch <=max_epoch)
                partition_number = 3;
            end
                
            description = strcat('partition_', int2str(partition_number));
            description = strcat(description, '_');

            if sum(contains(condition, thin))
                condition = strcat(description, '_thin');
                condition = strcat('_', condition);
                condition = strcat(factor_name, condition);
                D.trials(onset).trial_order = thin_order;  
                thin_order = thin_order + 1;
            elseif sum(contains(condition, medium))
                condition = strcat(description, '_medium');
                condition = strcat('_', condition);
                condition = strcat(factor_name, condition);
                D.trials(onset).trial_order = med_order;  
                med_order = med_order + 1;
            elseif sum(contains(condition, thick))
                condition = strcat(description, '_thick');
                condition = strcat('_', condition);
                condition = strcat(factor_name, condition);
                D.trials(onset).trial_order = thick_order; 
                thick_order = thick_order + 1;
            else
                condition = 'N/A';     
                D.trials(onset).trial_order = -999; 
            end

            condition_names{onset} = condition;
            D.trials(onset).label = condition;
        end
               
    end
    
    condition_names = unique(cellfun(@num2str,condition_names,'uni',0));
    condition_names(ismember(condition_names,'N/A')) = [];
    save(file, 'D') 
end

%% ensure there is 20% of stim per bucket
%% ensure there is atleast 20% of stimulus type per bucket
function reject_participant = reject_particiapnt_based_on_bad_trials(file)
    thin_onsets = {'65411'; '65412'; '65413'; '65414'; '65415'; '65416';'65417'; '65418'; '65419'};
    med_onsets = {'65401'; '65402'; '65403'; '65404'; '65405'; '65406'; '65407'; '65408'; '65409'};
    thick_onsets = {'65391'; '65392'; '65393'; '65394'; '65395'; '65396'; '65397'; '65398'; '65399'};

    load(file); % loads the D object
    n_trials = size(D.trials);  
    n_trials = n_trials(2);
    
    thin = [];
    med = [];
    thick = [];
    
    thin_count = 1;
    med_count = 1;
    thick_count = 1;
    
    threshold = 0.80;
    
    for onset = 1:n_trials
        
        is_bad = D.trials(onset).bad;
        onset_type = D.trials(onset).events.value;
            
        if sum(contains(onset_type, thin_onsets))
            thin(thin_count) = is_bad;
            thin_count = thin_count + 1;
        elseif sum(contains(onset_type, med_onsets))
            med(med_count) = is_bad;
            med_count = med_count  + 1;
        elseif sum(contains(onset_type, thick_onsets))
            thick(thick_count) = is_bad;
            thick_count = thick_count + 1;
        end
        
    end
    
    percentage_thin = sum(thin)/thin_count;
    percentage_med = sum(med)/med_count;    
    percentage_thick = sum(thick)/thick_count;
    
    reject_participant = 0;
    
    if percentage_thin > threshold
        reject_participant = 1;
    end
    
    if percentage_thin > threshold
        reject_participant = 1;
    end
   
    if percentage_thin > threshold
        reject_participant = 1;
    end
    
end


%% ensure there is atleast 20% of stimulus type per bucket
function ft_reject_participant = ft_reject_particiapnt_based_on_bad_trials(postprocessed, raw)
    trial_info = raw.trialinfo;
    [original_n_occurence, original_conditions] = hist(trial_info,unique(trial_info));
    
    pp_trial_info = postprocessed.sampleinfo(:,3);
    [pp_original_n_occurence, pp_original_conditions] = hist(pp_trial_info,unique(pp_trial_info));

    % make sure we have a minimum number of trials
    if ismember(1, pp_original_n_occurence) || ismember(0, pp_original_n_occurence)
       reject_participant = 1;
       return;
    end
   
    % lost a condition due to postprocessing
    original_size = numel(original_conditions);
    pp_size = numel(pp_original_conditions);
    if original_size ~= pp_size
       reject_participant = 1;
       return;
    end
    
    % make sure there are 20% trial per condition
    [~, n_conditions] = size(pp_original_n_occurence);
    for i = 1:n_conditions
        original_condition = original_conditions(i);
        new_condition = pp_original_conditions(i);
        
        if original_condition == new_condition
            original_count = original_n_occurence(i);
            new_count = pp_original_n_occurence(i);
            if (new_count/original_count) < 0.20
                reject_participant = 1;
                return ;
            end
        end
    end
    
    reject_participant = 0;
    
end

%% converts the trials to a fieldtrip readable format
function new_trials = convert_to_fieldtrip_format(trials)
    new_trials = {};
    
    [~, ~, samples] = size(trials);

    for k=1:samples
        t = trials(:,:,k);
        new_trials{k} = t;
    end 
    
end