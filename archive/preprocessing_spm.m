%% preprocessing pipeline for SPM *** Author; Cihan Dogan
clear all
cd("C:\ProgramFiles\PhD");
    

%% Change these variables depending on what you would like to do.
main_path = 'C:\ProgramFiles\PhD\participant_';
analyse_partitions = 0;
analyse_erps = 0;
look_at_factors = 0;
factors_to_investigate = {};
onsets = [
    2,3,4,5,6,7,8
];

rating = {' '};

if analyse_partitions == 1
    file_save = 'partitions';
    n_participants = 39;
elseif analyse_erps == 1
    file_save = 'onsets';
    n_participants = 40;
else
    file_save = 'mean_intercept';
    n_participants = 40;
end

%% some house keeping
if look_at_factors ~= 1
    factors_to_investigate = {''};
end
number_of_onsets = size(onsets);
number_of_onsets = number_of_onsets(1);


%% main preprocessing loop
for factor_k = 1:length(factors_to_investigate)
    factor_name = factors_to_investigate{factor_k};
    for i = 1:numel(onsets)
        subset_onsets = onsets(i,:);
        for participant = 1:n_participants
            %% gets the onsets of interest
            [thin, med, thick, description] = get_onsets(subset_onsets, analyse_partitions);

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
                data_structure = strcat(data_structure, '_075_80Hz.mat');   
                file_main_path = strcat(participant_main_path, data_structure);
                save_path = strcat(participant_main_path, 'SPM_ARCHIVE\');
                cd(participant_main_path);

                %% this function updates the trial information so that you only
                % analyse the conditions of interest
                condition_names = label_data(thin, med, thick, file_main_path, analyse_partitions, rating{1});

                %% filter
                matlabbatch{1}.spm.meeg.preproc.filter.D = {file_main_path};
                matlabbatch{1}.spm.meeg.preproc.filter.type = 'fir';
                matlabbatch{1}.spm.meeg.preproc.filter.band = 'bandpass';
                matlabbatch{1}.spm.meeg.preproc.filter.freq = [0.1 30];
                matlabbatch{1}.spm.meeg.preproc.filter.dir = 'twopass';
                matlabbatch{1}.spm.meeg.preproc.filter.order = 5;
                matlabbatch{1}.spm.meeg.preproc.filter.prefix = strcat(save_path, 'f1');
                spm_jobman('run',matlabbatch)
                clear matlabbatch

                %% baseline
                data_structure = strcat('f1', data_structure);
                file_main_path = strcat(save_path, data_structure);
                matlabbatch{1}.spm.meeg.preproc.bc.D = {file_main_path};
                matlabbatch{1}.spm.meeg.preproc.bc.timewin = [-200 0];
                matlabbatch{1}.spm.meeg.preproc.bc.prefix = 'b1';
                spm_jobman('run',matlabbatch)
                clear matlabbatch

                %% threshold
                fname = strcat(save_path,'trial-level', {'_'}, file_save);
                fname = fname{1};
                
                data_structure = strcat('b1', data_structure);
                file_main_path = strcat(save_path, data_structure);
                matlabbatch{1}.spm.meeg.preproc.artefact.D = {file_main_path};
                matlabbatch{1}.spm.meeg.preproc.artefact.mode = 'reject';
                matlabbatch{1}.spm.meeg.preproc.artefact.badchanthresh = 0.5;
                matlabbatch{1}.spm.meeg.preproc.artefact.append = true;
                matlabbatch{1}.spm.meeg.preproc.artefact.methods.channels{1}.type = 'EEG';
                matlabbatch{1}.spm.meeg.preproc.artefact.methods.fun.threshchan.threshold = 100;
                matlabbatch{1}.spm.meeg.preproc.artefact.methods.fun.threshchan.excwin = 1000;
                matlabbatch{1}.spm.meeg.preproc.artefact.prefix = fname;
                spm_jobman('run',matlabbatch)
                clear matlabbatch
     
                %% determine whether we need to reject an entire particiapnt
                dir = strcat(fname, data_structure);
                reject_participant = reject_particiapnt_based_on_bad_trials(dir);
                 
                 if reject_participant == 1
                     disp('Rejected participant not enough trials after 100 uv rejection');
                     delete(dir);
                     delete(replace(dir, '.mat', '.dat'));
                     continue;
                 end

                %% average EEG for each condition
                fname = strcat(save_path, 'grand-avg', {'_'});
                fname = fname{1};
                matlabbatch{1}.spm.meeg.averaging.average.D = {dir};
                matlabbatch{1}.spm.meeg.averaging.average.userobust.standard = false;
                matlabbatch{1}.spm.meeg.averaging.average.plv = false;
                matlabbatch{1}.spm.meeg.averaging.average.prefix = fname;
                spm_jobman('run',matlabbatch)
                clear matlabbatch

                disp(strcat('Processed participant..',int2str(participant))); 
            end
        end
    end
end
%% return the desired onsets
function [thin, med, thick, description] = get_onsets(onsets, is_partitioned)
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
    
    if is_partitioned
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
function condition_names = label_data(thin, medium, thick, file, analyse_partitions, factor_name)
    load(file); % loads the D object
    n_trials = size(D.trials);
    n_trials = n_trials(2);
    count = 1;
    condition_names = {};
    
    if analyse_partitions == 0
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
            
            if ~condition_found == 1
                error('Condition not found...');
            end

            if sum(contains(condition, thin))
                condition = strcat(factor_name, '_thin');
            elseif sum(contains(condition, medium))
                condition = strcat(factor_name, '_medium');
            elseif sum(contains(condition, thick))
                condition = strcat(factor_name, '_thick');
            else
                condition = -999;
            end

            condition_names{onset} = condition;
            D.trials(onset).label = condition;        
            count = count + 1;
        end      
    else
        % new partition fn
        if numel(thin) > 1
            onsets_thin_REF = {'65412'; '65413'; '65414'; '65415'; '65416';'65417'; '65418'};
            onsets_medium_REF = {'65402'; '65403'; '65404'; '65405'; '65406'; '65407'; '65408'};
            onsets_thick_REF = {'65392'; '65393'; '65394'; '65395'; '65396'; '65397'; '65398'};
        else
            onsets_thin_REF = {'65412'};
            onsets_medium_REF = {'65402'};
            onsets_thick_REF = {'65392'};
        end
        first_onsets = {'65411', '65401', '65391'};
        
        partition_number = 1;
        trial_count = 0;
        all_trial_count = 0;
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
            
            if ~condition_found == 1
                error('Condition not found...');
            end
            
            if sum(contains(condition, first_onsets))
                trial_count = trial_count + 1;
                all_trial_count = all_trial_count + 1;
            end
                
            if trial_count > 18
                partition_number = partition_number + 1;
                 % for sanity checks
                trial_count = 1;
            end
                
            description = strcat('partition_', int2str(partition_number));
            description = strcat(description, '_');

            if sum(contains(condition, thin))
                condition = strcat(description, 'thin');
                condition = strcat('_', condition);
                condition = strcat(factor_name, condition);
            elseif sum(contains(condition, medium))
                condition = strcat(description, 'medium');
                condition = strcat('_', condition);
                condition = strcat(factor_name, condition);
            elseif sum(contains(condition, thick))
                condition = strcat(description, 'thick');
                condition = strcat('_', condition);
                condition = strcat(factor_name, condition);
            else
                condition = -999;     
            end

            condition_names{onset} = condition;
            D.trials(onset).label = condition;
        end
               
    end
    
    condition_names = unique(cellfun(@num2str,condition_names,'uni',0));
    condition_names(ismember(condition_names,'-999')) = [];
    save(file, 'D') 
end

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
    
    threshold = 0.20;
    
    for onset = 1:n_trials
        events = D.trials(onset).events;
        is_bad = D.trials(onset).bad;
        
        [~, rows] = size(events);
        for i = 1:rows
            onset_type = events(i).binlabel;
            if ~strcmp(onset_type, '""')
                condition_found = 1;
                break
            end
        end
            
        if ~condition_found == 1
            error('Condition not found...');
        end
            
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
    
    percentage_thin = 1-(sum(thin)/thin_count);
    percentage_med = 1-(sum(med)/med_count);    
    percentage_thick = 1-(sum(thick)/thick_count);
    
    reject_participant = 0;
    
    if percentage_thin < threshold
        reject_participant = 1;
    end
    
    if percentage_thick < threshold
        reject_participant = 1;
    end
   
    if percentage_med < threshold
        reject_participant = 1;
    end
    
end
