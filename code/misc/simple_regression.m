clear;
clc;
log_trnsfrm = 1;


headache = get_scores('headache');
headache_power = get_scores('power');

[independent, dependent] = align_participants_based_on_number(headache, headache_power);

lm = fitlm(independent, dependent);
plot(lm);
xlabel('Headache')
ylabel('PGI of Power of Headache');
title('Dependent: PGI of Power of Headache, Independent: Headache Factor Scores'); 
disp(lm);

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
        
    elseif strcmp(type, 'headache')
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
    end
end