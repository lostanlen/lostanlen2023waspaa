fs = 44100;
N = fs*2;

ansi = [25,31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000];

[g,a,fc,fsupp]= cqtfilters_ansi(fs,25,20000,3,N,'uniform','fc',ansi');
%c = ufilterbank(f,g,a);

figure
gf = filterbankfreqz(g,a,N,fs,'plot','linabs','posfreq');
title('CQT filters, Q = 3')

L = filterbanklength(N,a);
[A,B] = filterbankrealbounds(g,a,L);
bounds = [A,B];

% num_filters = length(gf(1,:))
% stride = a(1)

save('Documents/MATLAB/Murenn/Freqz/third_freqz.mat','gf')
save('Documents/MATLAB/Murenn/Freqz/third_centerfreq.mat','fc')
save('Documents/MATLAB/Murenn/Freqz/third_bandwidths.mat','fsupp')
save('Documents/MATLAB/Murenn/Freqz/third_framebounds.mat','bounds')