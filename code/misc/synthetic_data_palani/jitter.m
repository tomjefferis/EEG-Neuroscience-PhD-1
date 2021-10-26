%code to test out Howard's idea on computing ERP jitter 

close all; clear all; clc;

temp=load ('P300grand_ave.mat'); %ERP signal (grand averagedas template)
x=temp.P300grand_ave;

JIT=30; %max jitter amount
tmp_x=[zeros(1,JIT),x, zeros(1,JIT)]; %pad zeros in front and rear

Fs=250; %smapling rate
for i=1:35 
nse(i,:)=noise(250+2*JIT,1,Fs);
signal(i,:)=nse(i,:)+tmp_x(1,:);
end

ave_x=mean(signal); %just try averaging

figure;
subplot(2,1,1),plot(x);
subplot(2,1,2),plot(ave_x);


%%%%%%%%%% Now, with jitter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:35 
tmp=randi([-JIT JIT]); %jitter range
    for j=1+JIT:260
    jit_sig(i,j-JIT)=signal(i,j+tmp);
    end
end

ave_jit_x=mean(jit_sig); %just try averaging with jitter

figure;
subplot(3,1,1),plot(x);
subplot(3,1,2),plot(ave_x);
subplot(3,1,3),plot(ave_jit_x);

signalAnalyzer(ave_jit_x) %spectrogram
