
EEG_pre=load('E:\1科研\代码\脑电分析代码\Preprocessing\Acupuncture\\sub_eeg\sub16_pre_split.mat');
ntrial=15;
con_sub=3;%pre acu post
%dep_sub=24;%抑郁被试或者训练组
nchannel=3;
fs=250;%采样频率
n_point=2000;%脑电数据点数
% 
% for sub=1:con_sub
%      
%     %c=ones(ntrial,1)*2;%%%正常标签
%     if sub ==2
%         ntrial = 225;
%     else 
%         ntrial = 15;
%     end
%     con_set=zeros(ntrial*3,8);
%     con_set_mean = zeros(ntrial,8);
%     for trial=1:ntrial
%         EEG=EEG_pre.sub16_pre_split{sub}{trial};
%         tmp = tiqutz12(EEG,EEG);
%         con_set((trial*3-2):trial*3,:)= tmp;
%         % 计算平均值
%         con_set_mean(trial,:) = mean(tmp,1);
%     end
%     feat_sub{sub}=con_set;%three lead data 
%     if sub == 2 %acu 数据225个，每15个再次取平均值
%         con_set_mean_tmp = zeros(15,8);
%         for i =1:15
%             con_set_mean_tmp(i,:) = mean(con_set_mean((i*15-14):i*15,:),1);
%         end
%         feat_sub_mean{sub} = con_set_mean_tmp;
%     else 
%         feat_sub_mean{sub} = con_set_mean;
%     end
% 
% end
% save('E:\1科研\代码\脑电分析代码\Preprocessing\Acupuncture\\sub_eeg\sub16_split_feat.mat','feat_sub');
% save('E:\1科研\代码\脑电分析代码\Preprocessing\Acupuncture\\sub_eeg\sub16_split_feat_mean.mat','feat_sub_mean');


% %融合所有sub的数据
% for i=1:16
%    path = ['E:\1科研\代码\脑电分析代码\Preprocessing\Acupuncture\\sub_eeg\sub' num2str(i) '_split_feat.mat'];
%    data_sub{i} = load(path);
% end
% % 
data_sub_mean_pre = zeros(15,8); %15段数据，8种特征
data_sub_mean_acu = zeros(225,8); %15段数据，8种特征
data_sub_mean_post = zeros(15,8); %15段数据，8种特征
data_tmp_pre = zeros(15,15);
data_tmp_acu = zeros(225,15);
data_tmp_post = zeros(15,15);
acu_feat_sub_mean_delta = {};
for k = 1:3
    for j = 1:8
        for i=1:15
            %data_tmp(:,i) = data_sub{1,i}.feat_sub{1,k}(:,j); %第二个括号 {1，1}是pre {1,2}是acu
            data_tmp_pre(:,i) = data_sub{1,i}.feat_sub{1,1}(:,j); %第二个括号 {1，1}是pre {1,2}是acu
            data_tmp_acu(:,i) = data_sub{1,i}.feat_sub{1,2}(:,j); %第二个括号 {1，1}是pre {1,2}是acu
            data_tmp_post(:,i) = data_sub{1,i}.feat_sub{1,3}(:,j); %第二个括号 {1，1}是pre {1,2}是acu
        end
        %data_sub_mean(:,j) = mean(data_tmp,2);
        data_sub_mean_pre(:,j) = mean(data_tmp_pre,2);
        data_sub_mean_acu(:,j) = mean(data_tmp_acu,2);
        data_sub_mean_post(:,j) = mean(data_tmp_post,2);
    end
    acu_feat_sub_mean_delta{k} = data_sub_mean;
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