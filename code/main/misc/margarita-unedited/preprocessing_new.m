%% preprocessing pipeline for SPM *** Author; Cihan Dogan
clear matlabbatch
cd("C:\Users\marga\Desktop\Research Project\scripts\SPM");
clear all
clc
%% Change these variables depending on what you would like to do.
main_path = 'C:\Users\marga\Desktop\Research Project\Cihan code\Data for experiment\Visual stress\participant_';
analyse_partitions = 0;
only_averaging = 1;
look_at_factors = 0;
factors_to_investigate = {};
onsets = [
    2,3,4,5,6,7,8;
];
n_participants=40;

partition1_onsets=0;
partition2_onsets=0;
partition3_onsets=0;

%% some house keeping
if look_at_factors ~= 1
    factors_to_investigate = {''};
end
number_of_onsets = size(onsets);
number_of_onsets = number_of_onsets(1);


%% main preprocessing loop
for factor_k = 1:length(factors_to_investigate)
    factor_name = factors_to_investigate{factor_k};
    for i = 1:number_of_onsets
        subset_onsets = onsets(i,:);
        for participant = 1:n_participants
%             if ismember(participant, [6,13])
%                 continue; 
%             end
            %% gets the onsets of interest
            [thin, med, thick, description] = get_onsets(subset_onsets, analyse_partitions);
            
            %% are we looking at factors?
            if look_at_factors == 1
                [rating, continue_proc] = get_nausea_percentile(participant, factor_name);
                if continue_proc ~= 1
                   continue; 
                end
                description = strcat(rating{1}, {'_'}, description{1});
            else
                rating = {''};
            end

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
                cd(participant_main_path);

                %% this function updates the trial information so that you only
                % analyse the conditions of interest
                condition_names = label_data(thin, med, thick, file_main_path, analyse_partitions, rating{1});

%                 partition1_onsets = partition1_onsets+p1;
%                 partition2_onsets = partition2_onsets+p2;
%                 partition3_onsets = partition3_onsets+p3;                
                
                %% filter
                matlabbatch{1}.spm.meeg.preproc.filter.D = {file_main_path};
                matlabbatch{1}.spm.meeg.preproc.filter.type = 'butterworth';
                matlabbatch{1}.spm.meeg.preproc.filter.band = 'bandpass';
                matlabbatch{1}.spm.meeg.preproc.filter.freq = [0.1 80];
                matlabbatch{1}.spm.meeg.preproc.filter.dir = 'twopass';
                matlabbatch{1}.spm.meeg.preproc.filter.order = 2;
                matlabbatch{1}.spm.meeg.preproc.filter.prefix = 'f1';        
                spm_jobman('run',matlabbatch)
                clear matlabbatch

%                 %% baseline
%                 data_structure = strcat('f1', data_structure);
%                 file_main_path = strcat(participant_main_path, data_structure);
%                 matlabbatch{1}.spm.meeg.preproc.bc.D = {file_main_path};
%                 matlabbatch{1}.spm.meeg.preproc.bc.timewin = [-200 0];
%                 matlabbatch{1}.spm.meeg.preproc.bc.prefix = 'b1';
%                 spm_jobman('run',matlabbatch)
%                 clear matlabbatch

%                 %% time-frequency decomposition
%                 data_structure = strcat('b1', data_structure);
%                 file_main_path = strcat(participant_main_path, data_structure);
%                 matlabbatch{1}.spm.meeg.tf.tf.D = {file_main_path};
%                 matlabbatch{1}.spm.meeg.tf.tf.channels{1}.all = 'all';
%                 matlabbatch{1}.spm.meeg.tf.tf.frequencies = 5:80;
%                 matlabbatch{1}.spm.meeg.tf.tf.timewin = [-Inf Inf];
%                 matlabbatch{1}.spm.meeg.tf.tf.method.morlet.ncycles = 5;
%                 matlabbatch{1}.spm.meeg.tf.tf.method.morlet.timeres = 0;
%                 matlabbatch{1}.spm.meeg.tf.tf.method.morlet.subsample = 5;
%                 matlabbatch{1}.spm.meeg.tf.tf.phase = 0;
%                 matlabbatch{1}.spm.meeg.tf.tf.prefix = '';
%                 spm_jobman('run',matlabbatch)
%                 clear matlabbatch
%                 
%                 %% crop
%                 data_structure = strcat('tf_', data_structure);
%                 file_main_path = strcat(participant_main_path, data_structure);
%                 matlabbatch{1}.spm.meeg.preproc.crop.D = {file_main_path};
%                 matlabbatch{1}.spm.meeg.preproc.crop.timewin = [-100 800];
%                 matlabbatch{1}.spm.meeg.preproc.crop.freqwin = [-Inf Inf];
%                 matlabbatch{1}.spm.meeg.preproc.crop.channels{1}.all = 'all';
%                 matlabbatch{1}.spm.meeg.preproc.crop.prefix = 'p';
%                 spm_jobman('run',matlabbatch)
%                 clear matlabbatch
%                 
                %% threshold
                data_structure = strcat('f1', data_structure);
                file_main_path = strcat(participant_main_path, data_structure);
                matlabbatch{1}.spm.meeg.preproc.artefact.D = {file_main_path};
                matlabbatch{1}.spm.meeg.preproc.artefact.mode = 'reject';
                matlabbatch{1}.spm.meeg.preproc.artefact.badchanthresh = 0.5;
                matlabbatch{1}.spm.meeg.preproc.artefact.append = true;
                matlabbatch{1}.spm.meeg.preproc.artefact.methods.channels{1}.type = 'EEG';
                matlabbatch{1}.spm.meeg.preproc.artefact.methods.fun.threshchan.threshold = 100;
                matlabbatch{1}.spm.meeg.preproc.artefact.methods.fun.threshchan.excwin = 1000;
                matlabbatch{1}.spm.meeg.preproc.artefact.prefix = 't1';
                spm_jobman('run',matlabbatch)
                clear matlabbatch
%                 
%                 %% average EEG for each condition
%                 data_structure = strcat('t1', data_structure);
%                 file_main_path = strcat(participant_main_path, data_structure);
%                 matlabbatch{1}.spm.meeg.averaging.average.D = {file_main_path};
%                 matlabbatch{1}.spm.meeg.averaging.average.userobust.standard = false;
%                 matlabbatch{1}.spm.meeg.averaging.average.plv = false;
%                 matlabbatch{1}.spm.meeg.averaging.average.prefix = strcat('averaged_', description{1});
%                 spm_jobman('run',matlabbatch)
%                 clear matlabbatch

                
                %% check if we only want to average for ERPs this is quicker to run over the entire cohort
                if only_averaging == 1
                    disp("We are only going to average, so we will skip creating images for now...")
                    continue;
                end

                 %% baseline rescaling / time-frequency rescale
%                 onset_description =  strcat('averaged_', description{1});
%                 data_structure = strcat(onset_description, data_structure);
%                 file_main_path = strcat(participant_main_path, data_structure);
%                 matlabbatch{1}.spm.meeg.tf.rescale.D = {file_main_path};
%                 matlabbatch{1}.spm.meeg.tf.rescale.method.LogR.baseline.timewin = [-100 0];
%                 matlabbatch{1}.spm.meeg.tf.rescale.method.LogR.baseline.pooledbaseline = 0;
%                 matlabbatch{1}.spm.meeg.tf.rescale.method.LogR.baseline.Db = [];
%                 matlabbatch{1}.spm.meeg.tf.rescale.prefix = 'r';
%                 spm_jobman('run',matlabbatch)
%                 clear matlabbatch
%                 
                 %% determine whether we need to reject an entire participant

%                 data_structure = strcat('t1', data_structure);
%                 file_main_path = strcat(participant_main_path, data_structure);
%                 reject_participant = reject_participant_based_on_bad_trials(file_main_path);
% 
%                 if reject_participant == 1
%                     disp('Rejected participant not enough trials after 100 uv rejection');
%                     continue;
%                 end

%                  %% convert to SPM images
%                 onset_description =  strcat('averaged_', description{1});
%                 data_structure = strcat(onset_description, data_structure);
%                 file_main_path = strcat(participant_main_path, data_structure);
%                 X.D = file_main_path;
%                 X.mode = 'scalp x time';
%                 X.conditions = condition_names;
%                 X.channels = {'EEG'};
%                 X.timewin = [-200,3999]; 
%                 X.freqwin = [1 80];
%                 X.prefix = '';
%                 spm_eeg_convert2images(X)

%                 %% create the pattern glare images
%                 file_main_path = strcat(participant_main_path, data_structure);
%                 img_path = participant_main_path;
%                 img_folder = data_structure(1:end-4);
%                 full_img_path = strcat(img_path, img_folder);
% 
%                 cd(full_img_path);
% 
%                 if analyse_partitions ~= 1
% 
%                     condition_name = strcat('\condition_', rating{1});
%                     condition_name = strcat(full_img_path, condition_name);
% 
%                     thin_img_path = strcat(condition_name,'_8.nii');
%                     medium_path = strcat(condition_name, '_9.nii');
%                     thick_path = strcat(condition_name, '_10.nii');
%                     erp_images = {medium_path;thin_img_path;thick_path};
%                     expression = 'i1 - (i2+i3)/2';
%                     img_name = strcat(description{1}, {'_'}, int2str(participant));
%                     img_name = img_name{1};
% 
%                     matlabbatch{1}.spm.util.imcalc.input = erp_images;
%                     matlabbatch{1}.spm.util.imcalc.output = img_name;
%                     matlabbatch{1}.spm.util.imcalc.outdir = full_img_path';
%                     matlabbatch{1}.spm.util.imcalc.expression = expression;
%                     matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
%                     matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
%                     matlabbatch{1}.spm.util.imcalc.options.mask = 0;
%                     matlabbatch{1}.spm.util.imcalc.options.interp = 1;
%                     matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
%                     spm_jobman('run',matlabbatch)
%                     clear matlabbatch
%                 else
%                     partitions = {
%                         'partition_1_8.nii','partition_1_9.nii', 'partition_1_10.nii';
%                         'partition_2_8.nii','partition_2_9.nii', 'partition_2_10.nii';
%                         'partition_3_8.nii','partition_3_9.nii', 'partition_3_10.nii';
%                     };   
% 
%                     if look_at_factors == 1
%                        desc = rating{1};
%                        desc = strcat('condition_', desc, {'_'});
%                        desc = desc{1};
%                     else
%                         desc = 'condition__';
%                     end
% 
%                     shape = size(partitions);
% 
%                     for partition_idx_i = 1:shape(1,1)
%                         for partition_idx_j = 1:shape(1,2)
%                             partition_name = partitions(partition_idx_i,partition_idx_j);
%                             new_partition_name = strcat(desc, partition_name);
%                             partitions(partition_idx_i, partition_idx_j) = new_partition_name;
%                         end
%                     end
% 
% 
%                     for partition_idx = 1:3
%                         subset = partitions(partition_idx,:);
%                         thin = subset{1};
%                         med = subset{2};
%                         thick = subset{3};
% 
%                         thin_img_path = strcat(full_img_path, {'\'}, thin);
%                         medium_path = strcat(full_img_path, {'\'}, med);
%                         thick_path = strcat(full_img_path, {'\'}, thick);
% 
%                         erp_images = {medium_path{1};thin_img_path{1};thick_path{1}};
%                         expression = 'i1 - (i2+i3)/2';
%                         img_name = strcat(description{1}, {'_'}, {int2str(partition_idx)}, {'_'}, int2str(participant));
%                         img_name = img_name{1};
% 
%                         matlabbatch{1}.spm.util.imcalc.input = erp_images;
%                         matlabbatch{1}.spm.util.imcalc.output = img_name;
%                         matlabbatch{1}.spm.util.imcalc.outdir = full_img_path';
%                         matlabbatch{1}.spm.util.imcalc.expression = expression;
%                         matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
%                         matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
%                         matlabbatch{1}.spm.util.imcalc.options.mask = 0;
%                         matlabbatch{1}.spm.util.imcalc.options.interp = 1;
%                         matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
% 
%                         spm_jobman('run',matlabbatch)
%                         clear matlabbatch
%                     end
%                 end

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
       description = {'partitioned_onsets'}; 
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
            cond = D.trials(onset).events.value;
            cond = string(cond);

            if sum(contains(cond, thin))
                condition = strcat(factor_name, '_8');
            elseif sum(contains(cond, medium))
                condition = strcat(factor_name, '_9');
            elseif sum(contains(cond, thick))
                condition = strcat(factor_name, '_10');
            else
                condition = -999;
            end

            condition_names{onset} = condition;
            D.trials(onset).label = condition;        
            count = count + 1;
        end      
    else
        % new partition fn
        onsets_thin_REF = {'65412'; '65413'; '65414'; '65415'; '65416';'65417'; '65418'};
        onsets_medium_REF = {'65402'; '65403'; '65404'; '65405'; '65406'; '65407'; '65408'};
        onsets_thick_REF = {'65392'; '65393'; '65394'; '65395'; '65396'; '65397'; '65398'};
        
%         first_onsets = {'65411', '65401', '65391'};
%         
%         partition_number = 1;
%         trial_count = 0;
%         all_trial_count = 0;
        p1=0;p2=0;p3=0;
        for onset = 1:n_trials
            condition = D.trials(onset).events.value;
            epoch = D.trials(onset).events.bepoch;
            %e = D.trials(onset).events.epoch;
%             if sum(contains(condition, first_onsets))
%                 trial_count = trial_count + 1;
%                 all_trial_count = all_trial_count + 1;
%             end
%                 
%             if trial_count > 18
%                 partition_number = partition_number + 1;
%                  % for sanity checks
%                 trial_count = 1;
%             end
            if epoch <= 144 % (6*3)*9
                partition_number = 1;
                p1 = p1+1;
            elseif epoch<=288
                partition_number = 2;
                p2 = p2+1;
            else
                partition_number = 3;
                p3 = p3+1;
            end
                
            description = strcat('partition_', int2str(partition_number));
            description = strcat(description, '_');

            if sum(contains(int2str(condition), onsets_thin_REF))
                condition = strcat(description, '8');
                condition = strcat('_', condition);
                condition = strcat(factor_name, condition);
            elseif sum(contains(int2str(condition), onsets_medium_REF))
                condition = strcat(description, '9');
                condition = strcat('_', condition);
                condition = strcat(factor_name, condition);
            elseif sum(contains(int2str(condition), onsets_thick_REF))
                condition = strcat(description, '10');
                condition = strcat('_', condition);
                condition = strcat(factor_name, condition);
            else
                condition = -999;     
            end

            condition_names{onset} = condition;
            D.trials(onset).label = condition;
            %display(epoch)
            %display(e)
        end
        display(epoch)
%         display(p1)
%         display(p2)
%         display(p3)
               
    end
    
    condition_names = unique(cellfun(@num2str,condition_names,'uni',0));
    condition_names(ismember(condition_names,'-999')) = [];
    save(file, 'D') 
end

%% ensure there is atleast 20% of stimulus type per bucket
function reject_participant = reject_participant_based_on_bad_trials(file)
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
 
%% based on factors puts them into buckets
function [score,continue_proc] = get_nausea_percentile(participant_number, factor_of_interest)

    continue_proc = 0;
    score = 'NULL';

    if strcmp(factor_of_interest,'discomfort')
        ratings = [
        1,-0.264;2,0.4459;3,-0.49781;4,1.77666;5,-0.55638;
        6,0.87174;7,-0.68504;8,0.92835;9,-0.80581;10,-0.87505;
        11,0.39111;12,-0.76054; 13,-0.68987;14,1.60776;16,1.13956;
        17,1.53606;20,0.12186;21,0.08428;22,0.61663;23,-1.47958;
        24,2.28422;25,-0.80891;26,-0.55738;28,-0.93291;29,0.3791;
        30,-0.63074;31,2.14683;32,-1.49948;33,1.21954;34,-0.79734;
        37,-0.61345;38,-1.02592;39,-0.87653;40,0.444
        ];
    elseif strcmp(factor_of_interest,'discomfort-z-score-five')
        ratings = [
        31,2.14683;4,1.77666;14,1.60776;17,1.53606;33,1.21954;
        39,-0.87653;28,-0.93291;38,-1.02592;23,-1.47958;32,-1.49948;
        ];
    elseif strcmp(factor_of_interest,'discomfort-z-score-four')
        ratings = [
        31,2.14683; 4,1.77666; 14,1.60776; 17,1.53606;
        28,-0.93291; 38,-1.02592; 23,-1.47958; 32,-1.49948;
        ];
    elseif strcmp(factor_of_interest,'discomfort-z-score-three')
        ratings = [
        31,2.14683; 4,1.77666; 14,1.60776;
        38,-1.02592; 23,-1.47958; 32,-1.49948;
        ];
    elseif strcmp(factor_of_interest,'discomfort-z-score-two')
        ratings = [
        31,2.14683; 4,1.77666; 
        23,-1.47958; 32,-1.49948;
        ];
    elseif strcmp(factor_of_interest,'discomfort-z-score-one')
        ratings = [
        31,2.14683;
        32,-1.49948;
        ];
    elseif strcmp(factor_of_interest,'aura')
        ratings = [
        1,0.3227;2,-0.10861;3,-0.51018;4,1.1336;5,-0.63947;6,-1.21472;
        7,-0.33005;8,0.75238;9,-0.39025;10,-0.72205;11,-0.76904;12,-1.06297;
        13,-0.89853;14,-1.46715;16,-1.19871;17,0.15415;20,0.5867;21,1.0008;
        22,-0.11689;23,1.72091;24,0.14105;25,0.62214;26,-0.74829;28,0.67386;
        29,-0.02367;30,0.03638;31,0.6996;32,-0.29977;33,-0.64998;34,0.02624;
        37,0.79861;38,-0.58832;39,2.33323;40,2.26667 
        ];
        
    elseif strcmp(factor_of_interest,'headache')
        ratings = [
        1,-0.22667;2,-0.05198;3,-0.72116;4,0.53139;5,1.72021;6,-1.17636;
        7,-0.79706;8,0.19942;9,-0.6924;10,-0.35826;11,-0.58533;12,2.04136;
        13,-0.26573;14,1.30963;16,0.00172;17,-0.15026;20,-0.41626;21,1.14373;
        22,-0.56513;23,-0.72755;24,-0.43472;25,-0.69897;26,-1.34952;28,0.36296;
        29,0.04162;30,-0.4697;31,-0.70362;32,-0.73219;33,-0.88081;34,-0.79623;
        37,1.22665;38,-0.3365;39,1.07651;40,-0.16678
        ];
    end 

    data(:,1) = ratings(:,2);
    data(:,2) = ratings(:,1);
    data = flipud(sortrows(data));
    n = size(data);
    n = n(1);
    
  
    top = data(1:(n/2),:);
    bottom = data((n/2)+1:n,:);

    if ismember(participant_number, top(:,2))
        score = strcat(factor_of_interest, {'_'}, 'high');
        continue_proc = 1;
        return
    elseif ismember(participant_number, bottom(:,2))
        score = strcat(factor_of_interest, {'_'}, 'low');
        continue_proc = 1;
        return
    else
        continue_proc = 0;
    end    
end