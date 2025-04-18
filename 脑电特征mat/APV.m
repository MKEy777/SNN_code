%author:zlx
%data:2022.04.25
% APV
%可以用bandpower提取各个频段的信息
function apv_val=APV(data,x)
    Vr=0;
    N=length(data);
    for i =1:N
        Vr=data(i)^2+Vr;
    end
    Wi=Vr/N;
    Wo=mean(x);
   % print('ceshi=%f',Wi);
    
   Wt=0;
    for i=1:1
        
        Wt=(Wi(i)-Wo)^2+Wt;
    end
    delt=Wt/1;
    apv_val=delt/Wo;
end
