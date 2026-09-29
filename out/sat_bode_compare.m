function sat_bode_compare(csvFile, torqueLimit)
% SAT_BODE_COMPARE  Effect of torque-command saturation on the Bode curves
%   sat_bode_compare('ScanFreq_20260928-104750.csv')          % limit auto-detected
%   sat_bode_compare('ScanFreq_20260928-104750.csv', 0.4998)  % limit given
%
% Assumed channels (GUESS, edit below if wrong):
%   x1 = excitation / command   x2 = speed feedback   x3 = torque command (saturates)
% Sweep is a log chirp: f(t) = f1*(f2/f1)^(t/Tsum); Fs/Tsum/f1/f2 are read from the
% CSV header when present.

if nargin < 1, csvFile = 'ScanFreq_20260928-104750.csv'; end
NHDR = 36;                                   % header lines before the data
P = read_header(csvFile, NHDR);
d = dlmread(csvFile, ',', NHDR, 0);          % 160000 x 4
x1 = d(:,1); x2 = d(:,2); x3 = d(:,3);
Fs = getp(P,'Pjet.Fs',16000); Tsum = getp(P,'Pjet.Tsum',10);
f1 = getp(P,'Pjet.f1',5);     f2 = getp(P,'Pjet.f2',5000);
if nargin < 2, torqueLimit = max(abs(x3)); end
thr = 0.999*torqueLimit;                     % |x3| >= thr counts as clipped
t = (0:numel(x3)-1).'/Fs;
kk = log(f2/f1)/Tsum;
phase = 2*pi*f1*(exp(kk*t)-1)/kk;            % instantaneous chirp phase [rad]

fq = logspace(log10(6), log10(4500), 200).'; % analysis frequencies [Hz]
n = numel(fq);
X = zeros(n,3); sat = zeros(n,1);
for i = 1:n
    [a,b] = win_idx(fq(i), f1, f2, Tsum, Fs, numel(t));
    e  = exp(-1j*phase(a:b)); w = hann_win(b-a+1);
    X(i,1) = sum(x1(a:b).*w.*e);
    X(i,2) = sum(x2(a:b).*w.*e);
    X(i,3) = sum(x3(a:b).*w.*e);
    seg = x3(a:b);
    sat(i) = mean(abs(seg) >= thr);                       % clipped fraction
end

H31 = X(:,3)./X(:,1);   % x3/x1 : controller / sensitivity-like (affected by clipping)
H21 = X(:,2)./X(:,1);   % x2/x1 : closed loop                     (affected)
H23 = X(:,2)./X(:,3);   % x2/x3 : plant torque->speed             (uses the real, clipped torque)

% First-order correction of x3's fundamental: clipped sine -> describing function N(r)
Hc31 = H31; corrGain = ones(n,1);
for i = 1:n
    if sat(i) > 0.01
        r = sin((1-sat(i))*pi/2);                        % r = limit / true amplitude
        N = (2/pi)*(asin(r) + r*sqrt(1-r^2));            % fundamental gain of clipper
        corrGain(i) = N; Hc31(i) = H31(i)/N;
    end
end
band = sat > 0.01;
if any(band)
    fprintf('Saturated band: %.1f - %.1f Hz, max clipped fraction %.1f %%\n', ...
        min(fq(band)), max(fq(band)), 100*max(sat));
    fprintf('Max estimated x3/x1 magnitude error: %.2f dB at %.0f Hz\n', ...
        max(-20*log10(corrGain)), fq(find(-20*log10(corrGain)==max(-20*log10(corrGain)),1)));
else
    disp('No saturation detected.');
end

db = @(h) 20*log10(abs(h)); pd = @(h) 180/pi*unwrap(angle(h));
lo = min(fq(band)); hi = max(fq(band));
fig = figure('Position',[50 50 1100 900],'Name','Saturation impact on Bode');
subplot(4,1,1); plot(t, x3, 'b'); hold on; grid on
plot([0 Tsum],  torqueLimit*[1 1],'r--',[0 Tsum], -torqueLimit*[1 1],'r--');
xlabel('t [s]'); title('x3 (torque command) with limit');
subplot(4,1,2); semilogx(fq, 100*sat, 'r'); grid on
ylabel('%'); title('Clipped samples of x3 per frequency');
subplot(4,1,3); semilogx(fq, db(H31), 'b', fq, db(Hc31), 'r--', 'LineWidth',1.2); grid on; hold on
shade_band(lo,hi,band); ylabel('dB'); legend('x3/x1 measured','x3/x1 corrected (describing fn)');
title('Magnitude of x3/x1 : with vs without saturation effect');
subplot(4,1,4); semilogx(fq, db(Hc31)-db(H31), 'k'); grid on
ylabel('dB'); xlabel('Frequency [Hz]'); title('Estimated magnitude error caused by saturation');

fig2 = figure('Position',[80 80 1100 800],'Name','Bode (guessed channels)');
H = {H23, H21}; nm = {'x2/x3 plant (uses actual clipped torque)','x2/x1 closed loop (affected)'};
for k = 1:2
    subplot(2,2,k); semilogx(fq, db(H{k}), 'b'); grid on; hold on; shade_band(lo,hi,band);
    title(nm{k}); ylabel('dB');
    subplot(2,2,k+2); semilogx(fq, pd(H{k}), 'b'); grid on; hold on; shade_band(lo,hi,band);
    ylabel('deg'); xlabel('Frequency [Hz]');
end
if exist('OCTAVE_VERSION','builtin') == 0
    % MATLAB: figures stay open. nothing else to do
else
    print(fig,  '-dpng', 'sat_effect_octave.png'); print(fig2, '-dpng', 'bode_octave.png');
end
end

% ---------------- helpers ----------------
function P = read_header(fn, nh)
P = struct(); fid = fopen(fn,'r');
for i = 1:nh
    s = fgetl(fid); if ~ischar(s), break; end
    c = strsplit(s, ',');
    if numel(c) >= 2 && ~isempty(c{2}) && ~isnan(str2double(c{2}))
        P.(strrep(c{1},'.','_')) = str2double(c{2});
    end
end
fclose(fid);
end
function v = getp(P, name, dflt)
k = strrep(name,'.','_'); if isfield(P,k), v = P.(k); else, v = dflt; end
end
function [a,b] = win_idx(f0, f1, f2, Tsum, Fs, N)
tc = Tsum*log(f0/f1)/log(f2/f1);              % time when the chirp is at f0
w  = min(max(8/f0, 0.02), 0.3);               % ~8 cycles, 20 ms .. 300 ms
a = max(round((tc-w/2)*Fs)+1, 1); b = min(round((tc+w/2)*Fs), N);
end
function w = hann_win(m)
w = 0.5 - 0.5*cos(2*pi*(0:m-1).'/(m-1));
end
function shade_band(lo, hi, band)
if any(band)
    yl = ylim; plot([lo lo],yl,'r:',[hi hi],yl,'r:'); ylim(yl);   % saturated band edges
end
end
