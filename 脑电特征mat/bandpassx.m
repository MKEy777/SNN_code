function output=bandpassx(x,fs,fc1,fc2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% �ļ����ܣ�������ͨFIR�˲����������������н�С���԰�ͽϴ��˥���ٶȣ�����ʵ��ʵʱ��
% ���з�ʽ������output=bandpass_hanning(x,fs,fc1,fc2)��fc1��fc2�ֱ�Ϊ�˲���������Ƶ�ʱ߽磬fsΪ����Ƶ�ʣ�xΪ�˲����������źţ����Ϊoutput��
%�˳���ǰ�����ݻ�����ʧ�棬���ݳ�����ԭʼ���ݱ���һ�¡�
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
N=512;    %ͨ���߽�Ƶ�ʣ���һ��Ƶ�ʣ����˲�������

if a_1~=1 && b_1~=1
    error('MATLAB:bandpass_firls:Inputmatrixisnotreliable',...
              'Input matrix is not a one - dimensional array.  See bandpass_firls.');
end
if fc1<0.2 || fc2>512
    error('MATLAB:bandpass_firls: Input low frequency must bigger than 0.2Hz or the high frequency must lower than 512Hz! See bandpass_firls.m.');
end
b=fir1(N,wp,hanning(N+1));           %���FIR��ͨ�˲���
output1=conv(b,x);
M=floor((N-1)/2);
output=output1(M:M+len-1);

if updown==1
    output=output';
end
