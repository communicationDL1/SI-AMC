# SI-AMC
The implement of SI-AMC using keras and Matlab.
Note: This work has already been accepted by IEEE WCNC 2024 and the repository contains some of the implementation code for our research work. 
# Requirements
Matlab 2022a, keras=2.9.0 tensorflow-gpu=2.9.0, python 3.8+
# Scenario Dataset
run traindatasetgeneration.m to generater rather than traindata.m. traindata.m is used for generating your own dataset.
# Matlab code
1. Simulate IEEE 802.11p environment to generate datasets with varying signal-to-noise ratios (SNR) which ranges from 15 dB to 38 dB under different scenarios.
2. Adaptive Modulation and Coding
Note: maxNumErrors and maxNumPackets should be larger to avoid randomness.(eg: maxNumErrors and maxNumPackets are 100 and 1000 respectively!)
# Python code
Utilize Keras to train ECNN model for Scenario Identification.
# Paper
Pending publication！
