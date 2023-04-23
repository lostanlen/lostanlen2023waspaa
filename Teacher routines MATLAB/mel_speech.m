fs = 16000;
N = fs*2;

[g,a,fc,L,fsupp] = audfilters_1(fs,N,'uniform','mel','spacing',70);
% c = filterbank(f,g,a);

figure
gf = filterbankfreqz(g,a,L,fs,'plot','linabs','posfreq');
title('AUDlet filters on Mel-scale.')

[A,B] = filterbankrealbounds(g,a,L);
bounds = [A,B];

%num_filters = length(gf(1,:))
%stride = a(1)

save('Documents/MATLAB/Murenn/Freqz/mel_freqz.mat','gf')
save('Documents/MATLAB/Murenn/Freqz/mel_centerfreq.mat','fc')
save('Documents/MATLAB/Murenn/Freqz/mel_bandwidths.mat','fsupp')
save('Documents/MATLAB/Murenn/Freqz/mel_framebounds.mat','bounds')