%%%%%%%%%%%%%%%%%%%%%%shannon_entropy%%%%%%%%%%%%%%%%%%%%%
%%作者：李兰兰
%%日期：2010.07.02
%%该程序用于计算时间序列的Renyi熵谱，该熵是香农熵的一般形式，同时包含了信号的幅值和频率信息，它可以分析非平稳过程或者非高斯过程的时间序列，广泛
%%适用于物理问题，尤其是具有分形结构、长程相关和长时间记忆的系统。由于大脑是一个复杂的非线性系统，故Renyi熵谱可以作为分析脑电信号谱的一个恰当的
%%工具。
%%Renyi熵是一种较好的时频分析工具，一般来说信号的基本组成成分越单一，Renyi值越小，相反信号越复杂，组成成分越多，Renyi熵值越大。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%关于输入：该程序中A代表输入矩阵（单导数据），Fs为输入的采样率，p如果等于1则画出Renyi熵谱的曲线图
%关于输出：D_inf每一次计算得到的q趋于无穷大时的Renyi值，D_neinf代表每一次计算得到的q趋于无穷小时得到的Renyi值，D_q代表程序每次计算得到的Renyi特征值（q此时为一个取值区间），
%每4s个点计算一次，计算时每次重叠2s，其中q为计算Renyi值的阶数
%average_D_inf,average_D_neginf,Renyi_entropy分别为每次输出的D_inf,D_neginf,D_q的平均值
% D_inf
% D_neginf
% D_q_1
% D_q_0
function [D_inf_all,D_q_0all,D_q_1all,average_D_inf,average_D_q_0,average_D_q_1]=Renyi_spectral(A,Fs,p,window)
%A=load(input);
Len_signal=length(A);
%Fs=256;
window_t=window;
M=Fs*window_t;                            %每次计算的序列长度
slide_point=Fs*(window_t);                   %每次滑动的点数
h=floor((Len_signal-M)/slide_point);
for j=0:h
    Xt=A(1+j*slide_point:M+j*slide_point);
    delta_V=0.01;                      %脑电测量设备的精准度
    X_max=max(Xt);                     %待计算序列的最大幅值（即待计算脑电信号的最大电压值）
    X_min=min(Xt);                     %待计算序列的最小幅值
    N=fix((X_max-X_min)/delta_V);           %整个信号被划分的区间数
    Xt_axis=fix((Xt-X_min)/delta_V)+1; %计算信号中每个点的坐标区间值
    histograms=unique(Xt_axis);
    [K,Point]=size(histograms);
    P_Xt=zeros(Point,1);
    for i=1:Point
        for k=1:M
            if histograms(i)==Xt_axis(k)
                P_Xt(i)=P_Xt(i)+1;     %计算每个区间落入的点数
            end
        end
    end
    Pxt=P_Xt./M;                      %计算该序列的概率分布
    P_max=max(Pxt);                   %计算概率最大值
    P_min=min(Pxt);                   %计算概率最小值
    D_inf=log(P_max)./log(delta_V);   %计算当q趋于无穷大时，得到的Renyi熵值
    D_neginf=log(P_min)./log(delta_V);%当q趋于无穷小时，得到的Renyi熵值
    delta_D=D_neginf-D_inf;
    D_inf_all(j+1)=D_inf;
    D_neginf_all(j+1)=D_neginf;
    delta_D_all(j+1)=delta_D;
    q=-50:1:50;                       %给定一个q的取值域
    q_num=length(q);
    D_q=zeros(q_num,1);
    for i=1:q_num
        for k=1:Point
            %if ((q(i)~=1)&(q(i)~=0))
            %sum_pi_q(i)=log(sum(Pxt(k).^q(i)));
            sum_pi_q(i,k)=(Pxt(k).^q(i));
        end
        sum_q(i)=sum(sum_pi_q(i,:));
        if q(i)~=1
            D_q(i)=(1./(q(i)-1))*(log(sum_q(i))./log(delta_V)); 
            D_q_0=D_q(51);                                          %得到当q等于0时的Renyi熵谱值
            D_q_1=D_q(1);       %修改！！！！！暂时表示 q = -50的值
        else
            D_q(i)=-sum(Pxt.*log(Pxt))./(log(1/delta_V));           %得到当q等于1时的Renyi熵谱值，即序列的香农熵
%             D_q_1=D_q(i);
        end
    end
    M_D(:,j+1)=D_q;
    D_q_0all(j+1)=D_q_0;
    D_q_1all(j+1)=D_q_1;
    if p==1
        plot(q,D_q,'k-');
        xlabel('q'); ylabel('D_q');
        hold on;
    end
end
average_D_inf=mean(D_inf_all);
%average_D_neginf=mean(D_neginf_all);
%average_delta_D=mean(delta_D_all);
average_D_q_0=mean(D_q_0all);
average_D_q_1=mean(D_q_1all);
Renyi_entropy=mean(M_D');
if p==1
    
    plot(q,Renyi_entropy,'r-');
    hold on
end
 
   
