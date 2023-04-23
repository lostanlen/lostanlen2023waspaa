fs = 16000;
N = fs*2;

a = 8;
filterlength = N;
L = ceil(filterlength/a)*a;
M = ceil(freqtoerb(fs/2)*2);
fc = erbspace(0,fs/2,M);

[g,fsupp] = gammatonefir(fc,fs,filterlength);
% c = filterbank(f,g,a);

figure
gf = filterbankfreqz(g,a,L,fs,'plot','linabs','posfreq');
title('4th order Gammatone FIR filters on ERB-scale')

[A, B] = filterbankrealbounds(g,a,L);

% num_filters = length(gf(1,:))


save('Documents/MATLAB/Murenn/Freqz/gam_freqz.mat','gf')
save('Documents/MATLAB/Murenn/Freqz/gam_centerfreq.mat','fc')
save('Documents/MATLAB/Murenn/Freqz/gam_bandwidths.mat','fsupp')
save('Documents/MATLAB/Murenn/Freqz/gam_framebounds.mat','bounds')