function[FAA]=FAA(EEG)
      L = EEG(:,1);
      R = EEG(:,3);
      % Pick frequency range
      FREQ_1 = 8;
      FREQ_2 = 13;
      % Compute power spectrum for left channel
      WIND = hamming(floor(length(L))/2);   % Changed to 2 was originally 1.5 sec time windows
      OVER = floor((length(L))/1.5/2);        % 50% overlap: previously /1.5/2
      SIGN = L';                              % Get signal
      [s, freqs, t, power] = spectrogram(SIGN, WIND, OVER, [], 250);
      indFreqs = find(freqs>FREQ_1 & freqs<FREQ_2);
      POW_L = power(indFreqs);
      % Compute power spectrum for right channel
      WIND = hamming(floor(length(R))/2);   % Get 1.5 sec time windows - changed to 2
      OVER = floor((length(R))/1.5/2);        % 50% overlap: previously /1.5/2
      SIGN = R';                              % Get signal
      [s, freqs, t, power] = spectrogram(SIGN, WIND, OVER, [],250);
      indFreqs = find(freqs>FREQ_1 & freqs<FREQ_2);
      POW_R = power(indFreqs);
      % Compute whole FAA
%       FAA = mean(abs(log(POW_R)-log(POW_L)));
      FAA = mean(log(POW_R)-log(POW_L));
end