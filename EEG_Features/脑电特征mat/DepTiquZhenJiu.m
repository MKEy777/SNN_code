ntrial=8;
con_sub=108;%正常被试或者对照组
%dep_sub=24;%抑郁被试或者训练组
nchannel=3;
fs=250;%采样频率
n_point=500;%脑电数据点数
%control=nc_sti;
%depression=md_sti;
for sub=1:con_sub
     
    %c=ones(ntrial,1)*2;%%%正常标签
    con_set=zeros(ntrial,7);
    for trial=1:ntrial
        EEG=EEG_Normal_Data{sub}{trial};
        con_set(trial,:)= tiqutz12(EEG);
        feat_Normal_0401{sub}=con_set;
     end
end

% for sub=1:dep_sub
%    
%     %a=ones(ntrial,1);%%%抑郁标签
%     dep_set=zeros(ntrial,nchannel*5);
%     
%     for trial=1:ntrial
%         EEG=depression{sub}{trial};
%         dep_set(trial,:)= tiqutz128(EEG);
%         feat_dep{sub}=[a,dep_set];
%     end
% end