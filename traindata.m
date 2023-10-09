%"""
%Created on Mon Oct 9 14:50:49 2023
% @author: Gray
clc
close all
clear all
% Link parameters
mcs = 1;       % 
psduLen = 1000; % PSDU length in bytes
channeltype=["Rural LOS","Urban approaching LOS","Urban NLOS","Highway LOS","Highway NLOS"];
NC=10000;
H = [];
for j=1:5
    for i = 1:NC
    % Create a format configuration object for an 802.11p transmission
    cfgNHT = wlanNonHTConfig;
    cfgNHT.ChannelBandwidth = 'CBW10';
    cfgNHT.PSDULength = psduLen;
    cfgNHT.MCS = mcs;
    fs = wlanSampleRate(cfgNHT); % Baseband sampling rate for 10 MHz

    chan = V2VChannel;
    chan.SampleRate = fs;
    chan.DelayProfile = channeltype(j);
    %You can choose an appropriate range of SNR for vehicular 
    temp = v2vChanEstSimulator(cfgNHT, chan, snr);
    H(i+NC*(j-1),:,1)= real(temp);
    H(i+NC*(j-1),:,2)= imag(temp);
    end
end
save('V2V_H_SI.mat','H')
    
