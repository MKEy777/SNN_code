ntrial=8;
con_sub=108;%�������Ի��߶�����
%dep_sub=24;%�������Ի���ѵ����
nchannel=3;
fs=250;%����Ƶ��
n_point=500;%�Ե����ݵ���
%control=nc_sti;
%depression=md_sti;
for sub=1:con_sub
     
    %c=ones(ntrial,1)*2;%%%������ǩ
    con_set=zeros(ntrial,7);
    for trial=1:ntrial
        EEG=EEG_Normal_Data{sub}{trial};
        con_set(trial,:)= tiqutz12(EEG);
        feat_Normal_0401{sub}=con_set;
     end
end

% for sub=1:dep_sub
%    
%     %a=ones(ntrial,1);%%%������ǩ
%     dep_set=zeros(ntrial,nchannel*5);
%     
%     for trial=1:ntrial
%         EEG=depression{sub}{trial};
%         dep_set(trial,:)= tiqutz128(EEG);
%         feat_dep{sub}=[a,dep_set];
%     end
% end