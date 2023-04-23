fs = 44100;
N = fs*2;

gamma = 24.7*(2^(1/12)-1)/0.108;

[g,a,fc,~,info] = vqtfilters(fs,100,fs/2,12,N,'uniform','gam',gamma);
% c = filterbank(f,g,a);

figure
gf = filterbankfreqz(g,a,N,fs,'plot','linabs','posfreq');
title('VQT filters, 12 bins per octave, gamma = 13.6')

L = filterbanklength(N,a);
[A,B] = filterbankrealbounds(g,a,L);
bounds = [A,B];

fsupp = info.fsupp;
% num_filters = length(gf(1,:))
% stride = a(1)

% % bandwidths
% loglog(fc(2:end-1),info.fsupp(2:end-1))

save('Documents/MATLAB/Murenn/Freqz/vqt_freqz.mat','gf')
save('Documents/MATLAB/Murenn/Freqz/vqt_centerfreq.mat','fc')
save('Documents/MATLAB/Murenn/Freqz/vqt_bandwidths.mat','fsupp')
save('Documents/MATLAB/Murenn/Freqz/vqt_framebounds.mat','bounds')