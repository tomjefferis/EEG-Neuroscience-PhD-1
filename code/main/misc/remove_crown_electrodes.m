%% given an image for each subject, get the ERP image
n_participants = 40;
main_path = 'D:\PhD\participant_';
clear matlabbatch

crown_electodes = {'A11', 'A12', 'A13', 'A14', 'A24', 'A25', 'A26','A27', 'B8', 'B9'};

for participant = 1:40
    clear matlabbatch
    disp(strcat('Procesisng participant..',int2str(participant)));
    participant_main_path = strcat(main_path, int2str(participant));
    if exist(participant_main_path, 'dir')
        data_structure = 'spmeeg_P';
        if participant < 10
            p = strcat('0', int2str(participant));
        else
            p = int2str(participant);
        end
        
        data_structure = strcat(data_structure, p);        
        data_structure = strcat(data_structure, '_075_80Hz_rejected.mat');
        file_main_path = strcat(participant_main_path, '\');
        file_main_path = strcat(file_main_path, data_structure);
        
        if isfile(file_main_path)
            load(file_main_path);
        else
            continue;
        end
        
        
        channel_data = D.channels;
        loop = size(channel_data);
        loop = loop(2);
        
        cnt = 0;
        for i=1:loop
            electrode = channel_data(i).label;
            if any(strcmp(crown_electodes,electrode))
               channel_data(i).type = 'Other';
            end        
        end
        
        D.channels = channel_data;
        save(file_main_path, 'D')
        
    end
end
