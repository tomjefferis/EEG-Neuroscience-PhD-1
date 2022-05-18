
load("D:\PhD\participant_2\SPM_ARCHIVE\partition_1_trial_level_5_80_Hz.mat");
spm = frequency_data;

load("D:\PhD\participant_2\SPM_ARCHIVE\fixed_partition_1_trial_level_5_80_Hz.mat");
ft = frequency_data;


spm_pwr = spm.med.powspctrm;
ft_pwr = ft.med.powspctrm;

sum(sum(sum(spm_pwr)), 'omitnan')
sum(sum(sum(ft_pwr)), 'omitnan')


-1.4106e+05 (ft preprocessing med)
-1.4106e+05

cnt = 0
for k=1:numel(med)
    cell = med(k);
    mtx = cell{1};
    cnt = cnt + sum(sum(sum(mtx)))
end


-8.1490e+05 ft data not spm


SPM powspectrm: 1.3885e+06 1.3885e+06


P1 pwscptrm FT: 2.9861e+05
p2 pwsctrm FT:  1.3885e+06

p1 pwsctrm SPM: 2.9861e+05 
p2 pwsctrm SPM: 1.1545e+06 1.3885e+06 (in isolation)    (1.1545e+06) after one another