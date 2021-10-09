clear;
clc;
log_trnsfrm = 1;

f = 'visual_stress ';

f = strcat(f, '-factor');
factor = get_scores(f);
itc = get_scores('itc');
power = get_scores('power');

%% create barplots
create_barplots(itc, factor)

%% create regressors
[~, y_itc] = align_participants_based_on_number(factor, itc);
[factors, y_power] = align_participants_based_on_number(factor, power);

factors = factors';
%factors(:,2) = 1;
power = y_power';
itc = y_itc';

factors_with_power = factors;
factors_with_power(:,3) = power(:,1);


% model one
tbl = table(factors, itc, 'VariableNames',...
    {'Factors_Through_Partitions','ITC_Through_Partitions'});
mdl = fitlm(tbl, 'ITC_Through_Partitions~Factors_Through_Partitions');
disp(mdl);

% model two
tbl = table(factors, power, 'VariableNames',...
    {'Factors_Through_Partitions','Power_Through_Partitions'});
mdl = fitlm(tbl, 'Power_Through_Partitions~Factors_Through_Partitions');
disp(mdl);


% model three
tbl = table(factors, power, itc, 'VariableNames',...
    {'Factors_Through_Partitions','Power_Through_Partitions', 'ITC_Through_Partitions'});
mdl = fitlm(tbl, 'ITC_Through_Partitions~Factors_Through_Partitions+Power_Through_Partitions');
disp(mdl);

%% create barplots
function create_barplots(itc, factor)
    fields = fieldnames(factor);
    
    all_high = [];
    all_low = [];
    for k = 1:numel(fields)
       partition_factor = factor.(fields{k}); 
       itc_factor = itc.(fields{k});
       
       [~,idx] = sort(partition_factor(:,2), 'descend');
       sortedmat = partition_factor(idx,:);
       participants_high = sortedmat(1:17,1);
       participants_low = sortedmat(18:34,1);
       
       low_cnt = 0;
       for n = 1:numel(participants_low)
          part_n =  participants_low(n);
          [row, ~] = find(itc_factor(:,1)==part_n);
          val = itc_factor(row,2);
          if sum(val) ~= 0
            low_cnt = low_cnt + val;
          end
       end
       
       high_cnt = 0;
       for n = 1:numel(participants_high)
          part_n =  participants_high(n);
          [row, ~] = find(itc_factor(:,1)==part_n);
          val = itc_factor(row,2);
          if sum(val) ~= 0
            high_cnt = high_cnt + val;
          end
       end
      
       
       all_low(k) = low_cnt;
       all_high(k) = high_cnt;
       
    end
    
    bar(all_low)
    ylim([-0.5, 1])
    title('Low Group: ITC Through the Partitions');
    bar(all_high)
    ylim([-0.5, 1])
    title('High Group: ITC Through the Partitions');
end

%% align participants
function [X, Y] = align_participants_based_on_number(x1, x2)
    fields = fieldnames(x1);
    
    [X,Y] = deal([], []);
    for k = 1:numel(fields)
        x1_partition = x1.(fields{k});
        x2_partition = x2.(fields{k});
        
        x1_n_participants = size(x1_partition,1);
        for p = 1:x1_n_participants
           x1_participant_number =  x1_partition(p,1);
           [row, ~] = find(x2_partition==x1_participant_number);
           x2_participant_number = x2_partition(row, 1);
           
           if numel(row) ~= 0 && x1_participant_number == x2_participant_number
              x2_score = x2_partition(row, 2);
              x1_score = x1_partition(p,2);
              
              X(end+1) = x1_score;
              Y(end+1) = x2_score;
           end
           
        end
    end
end

%% get the scores of interest
function scores = get_scores(type)
    if strcmp(type, 'power')
            scores.one = [
                1,0.515;2,-2.196;3,13.698;4,3.145;5,-2.199;6,-1.12;7,-0.187;
                8,6.547;9,0.143;10,1.268;11,3.298;12,1.823;14,13.738;16,-5.663;
                17,15.431;20,-0.562;21,3.445;22,1.882;23,-0.038;24,1.187;26,-8.355;
                28,-1.25;29,-1.307;30,2.316;31,-0.614;32,-0.5;33,-1.279;36,-1.38;
                37,-7.778;38,6.197;39,5.295;
            ];

            scores.two = [
                1,-1.549;2,-6.959;3,-3.575;4,-0.162;5,0.39;6,-0.704;7,0.173;8,2.999;
                9,1.954;10,0.113;11,-1.214;12,-0.392;14,12.759;16,-3.743;17,7.915;20,2.026;
                21,3.974;22,1.515;23,-0.474;24,18.276;26,-7.733;28,-0.185;29,-2.4;30,4.124;
                31,-6.307;32,-0.234;33,0.248;36,-0.482;37,22.504;38,-3.472;39,-2.79;
            ];

            scores.three = [
                1,-2.799;2,3.66;3,3.965;4,-8.609;5,-7.419;6,-1.419;7,-0.155;
                8,-1.799;9,-2.868;10,-2.614;11,0.997;12,-1.304;14,4.627;16,0.499;
                17,-13.738;20,-1.092;21,-5.965;22,1.744;23,0.759;24,-4.341;26,-2.607;28,1.316;29,0.051;
                30,4.519;31,-2.434;32,3.01;33,-1.068;36,-2.51;37,-2.409;38,3.922;39,-8.562;
            ];
       
    elseif strcmp(type, 'itc')
        scores.one = [
        1,0.277;2,-0.163;3,0.304;4,-0.053;5,0.042;6,0.002;7,0.063;8,0.133;
        9,0.058;10,0.095;11,0.022;12,-0.032;14,0.114;16,-0.052;17,0.107;
        20,0.076;21,0.049;22,0.311;23,-0.224;24,-0.219;26,-0.046;28,-0.046;
        29,-0.157;30,0.068;31,0.042;32,0.01;33,0.171;36,-0.167;37,0.295;38,-0.049;
        39,0.07;
        ];
        
        scores.two = [   
        1,0.193;2,0.051;3,0.116;4,0.03;5,-0.006;6,0.054;7,0.135;8,0.129;9,-0.064;
        10,-0.041;11,-0.064;12,-0.164;14,0.104;16,0.084;17,0.285;20,-0.031;21,0.197;
        22,0.17;23,-0.261;24,-0.101;26,0.018;28,-0.018;29,-0.009;30,0.214;31,-0.1;32,-0.052;33,-0.117;
        36,-0.097;37,0.143;38,-0.044;39,0.06;
        ];
    
        scores.three = [
        1,0.094;2,-0.111;3,0.274;4,-0.066;5,0.221;6,0.038;7,-0.036;8,0.081;
        9,-0.17;10,-0.247;11,-0.013;12,-0.156;14,0.148;16,-0.079;17,0.219;20,0.239;
        21,-0.285;22,0.172;23,-0.151;24,0.12;26,-0.081;28,0.011;29,-0.044;30,0.485;
        31,-0.028;32,-0.093;33,-0.04;36,0.242;37,-0.165;38,-0.087;39,-0.111;
        ];
        
    elseif strcmp(type, 'headache-factor')
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
        
        scores.one(:,2) = scores.one(:,2) * 2.72;
        scores.two(:,2) = scores.two(:,2) * 1.65;
        scores.three(:,2) = scores.three(:,2) * 1.00;
   
        [n_participants, ~] = size(dataset);
        
        for k=1:n_participants
            p1 = scores.one(k,1);
            p2 = scores.two(k,1);
            p3 = scores.three(k,1);
            
            if p1 == p2 && p2 == p3                
                to_remove = scores.three(k,2);
                
                scores.one(k,2) = scores.one(k,2) - to_remove;
                scores.two(k,2) = scores.two(k,2) - to_remove;
                scores.three(k,2) = scores.three(k,2) - to_remove;
            else
                error('Participants do not align...')
            end
        end   
    elseif strcmp(type, 'discomfort-factor')
        dataset = [
            1,-0.2427;2,0.4398;3,-0.5221;4,1.8399;5,-0.6095;6,0.8092;7,-0.6979;
            8,0.9717;9,-0.8232;10,-0.9152;11,0.3501;12,-0.8418;13,-0.7414;
            14,1.5086;16,1.0678;17,1.5466;20,0.1606;21,0.1343;22,0.6145;23,-1.3703;
            24,2.2964;25,-0.7656;26,-0.5905;28,-0.8957;29,0.3773;30,-0.6245;31,2.1948;
            32,-1.5111;33,1.1882;34,-0.7889;37,-0.5762;38,-1.0582;39,-0.7461;40,0.5811;
        ];
    
        scores.one = dataset;
        scores.two = dataset;
        scores.three = dataset;
    
        min_n = min(scores.one);
        scores.one(:,2) = scores.one(:,2) - min_n(2);
        scores.two(:,2) = scores.two(:,2) - min_n(2);
        scores.three(:,2) = scores.three(:,2) - min_n(2);
        
        scores.one(:,2) = scores.one(:,2) * 2.72;
        scores.two(:,2) = scores.two(:,2) * 1.65;
        scores.three(:,2) = scores.three(:,2) * 1.00;
   
        [n_participants, ~] = size(dataset);
        
        for k=1:n_participants
            p1 = scores.one(k,1);
            p2 = scores.two(k,1);
            p3 = scores.three(k,1);
            
            if p1 == p2 && p2 == p3                
                to_remove = scores.three(k,2);
                
                scores.one(k,2) = scores.one(k,2) - to_remove;
                scores.two(k,2) = scores.two(k,2) - to_remove;
                scores.three(k,2) = scores.three(k,2) - to_remove;
            else
                error('Participants do not align...')
            end
        end   
    elseif strcmp(type, 'visual_stress-factor')
        dataset = [
        1,0.3227;2,-0.1086;3,-0.5102;4,1.1336;5,-0.6395;6,-1.2147;
        7,-0.3301;8,0.7524;9,-0.3903;10,-0.7221;11,-0.769;12,-1.063;
        13,-0.8985;14,-1.4672;16,-1.1987;17,0.1542;20,0.5867;21,1.0008;
        22,-0.1169;23,1.7209;24,0.1411;25,0.6221;26,-0.7483;28,0.6739;
        29,-0.0237;30,0.0364;31,0.6996;32,-0.2998;33,-0.65;34,0.0262;
        37,0.7986;38,-0.5883;39,2.3332;40,2.2667;   
        ];
    
        scores.one = dataset;
        scores.two = dataset;
        scores.three = dataset;
    
        min_n = min(scores.one);
        scores.one(:,2) = scores.one(:,2) - min_n(2);
        scores.two(:,2) = scores.two(:,2) - min_n(2);
        scores.three(:,2) = scores.three(:,2) - min_n(2);
        
        scores.one(:,2) = scores.one(:,2) * 2.72;
        scores.two(:,2) = scores.two(:,2) * 1.65;
        scores.three(:,2) = scores.three(:,2) * 1.00;
   
        [n_participants, ~] = size(dataset);
        
        for k=1:n_participants
            p1 = scores.one(k,1);
            p2 = scores.two(k,1);
            p3 = scores.three(k,1);
            
            if p1 == p2 && p2 == p3                
                to_remove = scores.three(k,2);
                
                scores.one(k,2) = scores.one(k,2) - to_remove;
                scores.two(k,2) = scores.two(k,2) - to_remove;
                scores.three(k,2) = scores.three(k,2) - to_remove;
            else
                error('Participants do not align...')
            end
        end
    end
end