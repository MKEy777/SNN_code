function output=bandpassx(x,fs,fc1,fc2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 文件功能：基本带通FIR滤波器，汉宁窗，具有较小的旁瓣和较大的衰减速度，可以实现实时。
% 运行方式：输入output=bandpass_hanning(x,fs,fc1,fc2)，fc1、fc2分别为滤波器的上下频率边界，fs为采样频率，x为滤波器的输入信号，输出为output。
%此程序前端数据基本无失真，数据长度与原始数据保持一致。
%% @(#)$Id: bandpass_hanning.m 2010.12.30 Yanbing Qi Exp $ 0.1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check input signal
[a_1,b_1]=size(x);
updown=0;   %check the input data;
if b_1==1 && b_1<a_1
    x=x';
    updown=1;
end
[a_1,b_1]=size(x);
len=length(x);
wp=[2*fc1/fs 2*fc2/fs];
N=512;    %通带边界频率（归一化频率）和滤波器阶数

if a_1~=1 && b_1~=1
    error('MATLAB:bandpass_firls:Inputmatrixisnotreliable',...
              'Input matrix is not a one - dimensional array.  See bandpass_firls.');
end
if fc1<0.2 || fc2>512
    error('MATLAB:bandpass_firls: Input low frequency must bigger than 0.2Hz or the high frequency must lower than 512Hz! See bandpass_firls.m.');
end
b=fir1(N,wp,hanning(N+1));           %设计FIR带通滤波器
output1=conv(b,x);
M=floor((N-1)/2);
output=output1(M:M+len-1);

if updown==1
    output=output';
end
