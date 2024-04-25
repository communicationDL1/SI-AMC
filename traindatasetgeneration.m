clc;
close all;
clear all;
% Link parameters
mcs = 1;       % 
psduLen = 500; % PSDU length in bytes
channeltype=["Rural LOS","Urban approaching LOS","Urban NLOS","Highway LOS","Highway NLOS"];
NC=10000;
H = [];
%五种信道类型，每一种对应于10000种数据
for j=1:5
    for i = 1:NC
    % Create a format configuration object for an 802.11p transmission
    % WLAN 的配置对象
    cfgNHT = wlanNonHTConfig;
    cfgNHT.ChannelBandwidth = 'CBW10';
    cfgNHT.PSDULength = psduLen;
    cfgNHT.MCS = mcs;
    % Create and configure the channel
    %与自己设置的信道带宽保持一致
    fs = wlanSampleRate(cfgNHT); % Baseband sampling rate for 10 MHz
    %通过相应的信道模型
    chan = V2VChannel;
    chan.SampleRate = fs;
    chan.DelayProfile = channeltype(j);
    %估计采用不同的snr大小获得不同的场景数据,此处需要结合相应的信噪比对应调制方式的阈值
    snr =16;
    %信道估计
    temp = v2vChanEstSimulator(cfgNHT, chan, snr);
    H(i+NC*(j-1),:,1)= real(temp);
    H(i+NC*(j-1),:,2)= imag(temp);
    end
end
save('V2V_H_16.mat','H')
    