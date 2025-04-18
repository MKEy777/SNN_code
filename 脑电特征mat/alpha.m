function y = alpha(x)
%DOFILTER 对输入 x 进行滤波并返回输出 y。

% MATLAB Code
% Generated by MATLAB(R) 9.10 and DSP System Toolbox 9.12.
% Generated on: 24-Nov-2022 20:10:05

%#codegen

% 要通过此函数生成 C/C++ 代码，请使用 codegen 命令。有关详细信息，请键入 'help codegen'。

persistent Hd;

if isempty(Hd)
    
    % 设计滤波器系数时使用了以下代码:
    %
    % Fstop1 = 7;    % First Stopband Frequency
    % Fpass1 = 8;    % First Passband Frequency
    % Fpass2 = 13;   % Second Passband Frequency
    % Fstop2 = 14;   % Second Stopband Frequency
    % Astop1 = 60;   % First Stopband Attenuation (dB)
    % Apass  = 1;    % Passband Ripple (dB)
    % Astop2 = 80;   % Second Stopband Attenuation (dB)
    % Fs     = 250;  % Sampling Frequency
    %
    % h = fdesign.bandpass('fst1,fp1,fp2,fst2,ast1,ap,ast2', Fstop1, Fpass1, ...
    %                      Fpass2, Fstop2, Astop1, Apass, Astop2, Fs);
    %
    % Hd = design(h, 'butter', ...
    %     'MatchExactly', 'stopband', ...
    %     'SystemObject', true);
    
    Hd = dsp.BiquadFilter( ...
        'Structure', 'Direct form II', ...
        'SOSMatrix', [1 0 -1 1 -1.88675714188299 0.99312906407968; 1 0 -1 1 ...
        -1.95590981506791 0.995755990695216; 1 0 -1 1 -1.87431417728097 ...
        0.979604152504137; 1 0 -1 1 -1.94751080872906 0.987326727719593; 1 0 -1 ...
        1 -1.86260519734372 0.966486461381618; 1 0 -1 1 -1.93905880534311 ...
        0.97898354730635; 1 0 -1 1 -1.85174017449807 0.953908375083486; 1 0 -1 1 ...
        -1.93056349761583 0.97073672862167; 1 0 -1 1 -1.84181512043153 ...
        0.941991024064968; 1 0 -1 1 -1.92203547027427 0.962598479440698; 1 0 -1 ...
        1 -1.83291178530563 0.930843567118935; 1 0 -1 1 -1.91348715271276 ...
        0.954583821367509; 1 0 -1 1 -1.82509745065117 0.920562600841197; 1 0 -1 ...
        1 -1.90493380662201 0.946711478833158; 1 0 -1 1 -1.81842476446178 ...
        0.91123163662807; 1 0 -1 1 -1.89639455212134 0.939004767501573; 1 0 -1 1 ...
        -1.81293158655418 0.902920600781211; 1 0 -1 1 -1.88789342306065 ...
        0.931492463314171; 1 0 -1 1 -1.80864083630332 0.895685334188965; 1 0 -1 ...
        1 -1.87946042479577 0.924209615571709; 1 0 -1 1 -1.80556036279936 ...
        0.889567094518404; 1 0 -1 1 -1.87113254611134 0.917198247008819; 1 0 -1 ...
        1 -1.80368288760974 0.884592094880372; 1 0 -1 1 -1.86295465254928 ...
        0.910507862893282; 1 0 -1 1 -1.80298609928717 0.880771145892859; 1 0 -1 ...
        1 -1.85498016444755 0.904195673578917; 1 0 -1 1 -1.80343300112334 ...
        0.87809949811072; 1 0 -1 1 -1.84727140487605 0.898326426167821; 1 0 -1 1 ...
        -1.80497262236158 0.876557001805994; 1 0 -1 1 -1.83989949742954 ...
        0.892971747463151; 1 0 -1 1 -1.80754119092241 0.876108702825223; 1 0 -1 ...
        1 -1.83294370878019 0.888208927801052; 1 0 -1 1 -1.81106382791279 ...
        0.876705969618046; 1 0 -1 1 -1.82649017083241 0.884119125768203; 1 0 -1 ...
        1 -1.81545676173817 0.878288195106274; 1 0 -1 1 -1.82062998139741 ...
        0.880785043050331], ...
        'ScaleValues', [0.0638448332669975; 0.0638448332669975; ...
        0.0634951882163771; 0.0634951882163771; 0.0631551255481627; ...
        0.0631551255481627; 0.0628268353451944; 0.0628268353451944; ...
        0.0625123244924051; 0.0625123244924051; 0.0622134189432084; ...
        0.0622134189432084; 0.0619317682818188; 0.0619317682818188; ...
        0.061668851994449; 0.061668851994449; 0.0614259869174978; ...
        0.0614259869174978; 0.0612043353919514; 0.0612043353919514; ...
        0.0610049137159208; 0.0610049137159208; 0.0608286005483191; ...
        0.0608286005483191; 0.060676144973943; 0.060676144973943; ...
        0.0605481739923052; 0.0605481739923052; 0.060445199238821; ...
        0.060445199238821; 0.0603676227872814; 0.0603676227872814; ...
        0.0603157419172469; 0.0603157419172469; 0.0602897527596889; ...
        0.0602897527596889; 1]);
end

s = double(x);
y = step(Hd,s);

