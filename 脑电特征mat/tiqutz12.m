function [featureSet]=tiqutz12(EEG)
%printf('128');
feat_dim=7;
%EEG=control{1}{1};
if ~isempty(EEG)
[len,wei]=size(EEG);
if len<wei
   EEG=EEG';
  % wei=len;
end
end
[len,wei]=size(EEG);
featureSet=zeros(1,feat_dim);
featureSet_tmp=zeros(wei,feat_dim);
 if len==500
    z=zeros(12,1);
    else
        z=zeros(36,1);
 end
 len_size=len/250;

 
for chn=1:wei
   
    user_fp=[EEG(:,chn);z];
    [c01,c0_average]=c0complex(EEG(:,chn),250,0, len_size);
    %[Correlation1,M_C]=reC(EEG(:,chn),250,0, len_size);
    [D_inf_all,D_q_0all,D_q_1all1,average_D_inf,average_D_q_0,average_D_q_1]=Renyi_spectral(EEG(:,chn)',250,0, len_size);
    [Pxx1,F1] = fftpsd1(user_fp,250);

    ct1=center(Pxx1,F1);
    max1=max(Pxx1);
    mean1=sum(Pxx1)/129;
    EEG_band=alpha(EEG(:,chn));
%     EEG_band = EEGRhythm_alpha(EEG(:,chn));
%     apv=APV(EEG_band(400:2000,:),EEG_Orig(:,chn));
    faa = FAA(EEG);
    lzc=LZC(EEG(:,chn));
    r=0.2*std(EEG(:,chn));
    SEn = SampEn(2,r,EEG(:,chn),1);
%     feat=[c01,D_q_1all1,ct1,max1,mean1,faa,lzc,SEn];%[c01,Correlation1,D_q_1all1,ct1,max1,mean1];
    feat=[faa,ct1,max1,mean1,lzc,SEn,D_q_1all1];%[c01,Correlation1,D_q_1all1,ct1,max1,mean1];
    featureSet1(1,(chn-1)*feat_dim+1:chn*feat_dim)=feat;
    featureSet_tmp(chn,:)=feat;
%     featureSet=featureSet+feat;
    featureSet=featureSet_tmp;
end
end
%%%%%%%%%%%%-----------计算 频段  alpha beta

function [eeg_rhy_alpha]= EEGRhythm_alpha(x)
addpath(genpath('../Preprocessing/MIF-main/'));
% addpath(genpath('Acupuncture/'))
% EEG_rhy=cell(1,73);
% for i =1 : 73
    


% x = EEG1{1,i}'; % Take any multi-channel EEG signal in x
x = x';
Fs = 250;
opt=Settings_IF_v1('IF.Xi',1.6,'IF.alpha','ave','IF.delta',.001,'IF.NIMFs',20);
MIMF = IterFiltMulti(x,opt);
eeg_rhy_alpha = EEGRhythm(MIMF,Fs);
% EEG_rhy{1,i} = eeg_rhy;
% end
end


function [eeg_rhy]= EEGRhythm(mIMF,Fs)


N=size(mIMF{1,2},2); % Length of the data
N_ch=size(mIMF{1,1},1); % number of channel
N_mimf = size(mIMF,2); % Number of MIMF

for imf_ind = 1: N_mimf
    mf(:,imf_ind)=meanfreq(mIMF{1,imf_ind}', Fs);
end
mf = mean(mf,1);

delta_i=[];theta_i=[];alpha_i=[];
beta_i=[];gamma_i=[];
for i=1:length(mf) % Select IMF belongs to band
    if ((mf(i)>.1) && (mf(i) <4))
        delta_i=[delta_i,i];
    elseif (mf(i)>=4 && mf(i) <8)
        theta_i=[theta_i,i];
    elseif (mf(i)>=8 && mf(i) <14)
        alpha_i=[alpha_i,i];
    elseif (mf(i)>=14 && mf(i) <30)
        beta_i=[theta_i,i];
    elseif (mf(i)>=30 && mf(i) <95)
        gamma_i=[gamma_i,i];
    end
end

eeg_rhy{1,1} = EEG_Rhy(mIMF, delta_i);
eeg_rhy{1,2} = EEG_Rhy(mIMF, theta_i);
eeg_rhy{1,3} = EEG_Rhy(mIMF, alpha_i);
eeg_rhy{1,4} = EEG_Rhy(mIMF, beta_i);
eeg_rhy{1,5} = EEG_Rhy(mIMF, gamma_i);


end


% Axulary functions

function [Rhy] = EEG_Rhy(mIMF, ind)
Rhy = zeros(size(mIMF{1,1}));
for i = 1: length(ind)
    Rhy = Rhy + mIMF{1,ind(i)};
end
end



%featureSet=featureSet+feat;