%% ============================================================
%  GRIT SIMULATION -- UNIFORM GATE = Beta(1,1)
%  CLEAN VERSION FROM SCRATCH
%
%  Four signals:
%     Blocks      -> Haar
%     Bumps       -> Daubechies 3 (6-tap)
%     Doppler     -> Symmlet 4 (8-tap)
%     HeaviSine   -> Symmlet 4 (8-tap)
%
%  SNR definition:
%
%       SNR = E(signal^2) / Var(noise)
%
%  Noise variance is fixed at 1, so each signal is rescaled
%  separately so that
%
%       mean(signal.^2) = SNR
%
%  for SNR = 3, 5, 7.
%
%  GRIT gate:
%       Uniform = Beta(1,1), F(u)=u
%
%  GRIT:
%       delta(d)=0,                                  |d| <= lambda
%       delta(d)=d*[1-(lambda/|d|)^p],               |d| > lambda
%
%  p = 0:0.1:10
%
%  Known sigma = 1:
%       lambda = sqrt(2*log(n))
%
%  AMSE:
%       mean over Monte Carlo replicates of
%       mean((estimate-true_signal).^2)
%
%  IMPORTANT:
%  This version does NOT use the custom dwtr/idwtr functions.
%  It uses MATLAB Wavelet Toolbox wavedec/waverec with periodic
%  extension so transform/reconstruction are internally matched.
% ============================================================

clear;
clc;
close all;

rng(123);

%% ============================================================
% 1. SETTINGS
% =============================================================

n = 1024;
J = 5;
R = 1000;

SNRlist = [3 5 7];
signals = {'Blocks','Bumps','Doppler','HeaviSine'};

pGrid = 0:0.1:10;

sigma = 1;
lambda = sigma*sqrt(2*log(n));

nSignals = numel(signals);
nSNR = numel(SNRlist);
nP = numel(pGrid);

% Require MATLAB Wavelet Toolbox
if exist('wavedec','file') ~= 2 || exist('waverec','file') ~= 2
    error(['This script requires MATLAB Wavelet Toolbox functions ' ...
           'wavedec and waverec.']);
end

% Use periodic boundary handling, matching the usual periodic DWT setup.
oldMode = dwtmode('status','nodisp');
cleanupMode = onCleanup(@() dwtmode(oldMode,'nodisp'));
dwtmode('per','nodisp');

Results = struct();
Summary = table();

fprintf('\n=============================================\n');
fprintf('UNIFORM GRIT = Beta(1,1)\n');
fprintf('n = %d, J = %d, R = %d\n',n,J,R);
fprintf('noise variance = %.1f\n',sigma^2);
fprintf('lambda = %.6f\n',lambda);
fprintf('p grid = 0:0.1:10\n');
fprintf('=============================================\n');

%% ============================================================
% 2. SIGNAL LOOP
% =============================================================

for iSig = 1:nSignals

    signalName = signals{iSig};

    % ---------------------------------------------------------
    % Wavelet choice
    % ---------------------------------------------------------

    switch signalName
        case 'Blocks'
            wname = 'haar';

        case 'Bumps'
            wname = 'db3';       % 6-tap Daubechies

        case {'Doppler','HeaviSine'}
            wname = 'sym4';      % 8-tap Symmlet

        otherwise
            error('Unknown signal: %s',signalName);
    end

    % ---------------------------------------------------------
    % Generate the ORIGINAL test signal internally
    % ---------------------------------------------------------

    yBase = make_test_signal(signalName,n);
    yBase = yBase(:)';

    basePower = mean(yBase.^2);

    if basePower <= 0
        error('Nonpositive base signal power for %s.',signalName);
    end

    %% ========================================================
    % 3. SNR LOOP
    % =========================================================

    for iSNR = 1:nSNR

        SNRvalue = SNRlist(iSNR);

        % -----------------------------------------------------
        % RESCALE SIGNAL:
        %
        % SNR = mean(signal.^2)/sigma^2
        % sigma^2 = 1
        %
        % therefore force mean(yTrue.^2)=SNRvalue exactly.
        % -----------------------------------------------------

        scaleFactor = sqrt(SNRvalue/basePower);
        yTrue = scaleFactor*yBase;

        signalPower = mean(yTrue.^2);
        achievedSNR = signalPower/(sigma^2);

        if abs(achievedSNR-SNRvalue) > 1e-12
            error('SNR scaling failed for %s, SNR=%g.', ...
                  signalName,SNRvalue);
        end

        % -----------------------------------------------------
        % Transform/reconstruction sanity check
        % -----------------------------------------------------

        [cTrue,bookTrue] = wavedec(yTrue,J,wname);
        yCheck = waverec(cTrue,bookTrue,wname);
        yCheck = yCheck(:)';

        reconError = max(abs(yCheck-yTrue));

        if reconError > 1e-8
            error(['wavedec/waverec reconstruction failed for %s, ' ...
                   'SNR=%g. Max error %.3e'], ...
                  signalName,SNRvalue,reconError);
        end

        % Coefficient layout:
        % [A_J | D_J | D_{J-1} | ... | D_1]
        nApprox = bookTrue(1);
        detailIdxTrue = (nApprox+1):numel(cTrue);

        trueApproxEnergy = sum(cTrue(1:nApprox).^2)/n;
        trueDetailEnergy = sum(cTrue(detailIdxTrue).^2)/n;

        % -----------------------------------------------------
        % Storage
        % -----------------------------------------------------

        MSE_Noisy   = zeros(R,1);
        MSE_Coarse  = zeros(R,1);
        MSE_Soft    = zeros(R,1);
        MSE_Garrote = zeros(R,1);
        MSE_Hard    = zeros(R,1);

        MSE_GRIT = zeros(R,nP);

        %% ====================================================
        % 4. MONTE CARLO LOOP
        % =====================================================

        for r = 1:R

            % -------------------------------------------------
            % Add N(0,1) noise
            % -------------------------------------------------

            noise = randn(1,n);
            yNoisy = yTrue + noise;

            MSE_Noisy(r) = mean((yNoisy-yTrue).^2);

            % -------------------------------------------------
            % DWT
            % -------------------------------------------------

            [c,book] = wavedec(yNoisy,J,wname);

            nApproxThis = book(1);
            detailIdx = (nApproxThis+1):numel(c);

            d = c(detailIdx);
            ad = abs(d);

            % -------------------------------------------------
            % Coarse-only
            % -------------------------------------------------

            cCoarse = c;
            cCoarse(detailIdx) = 0;

            yCoarse = waverec(cCoarse,book,wname);
            yCoarse = yCoarse(:)';

            MSE_Coarse(r) = mean((yCoarse-yTrue).^2);

            % -------------------------------------------------
            % Soft threshold
            % -------------------------------------------------

            dSoft = sign(d).*max(ad-lambda,0);

            cSoft = c;
            cSoft(detailIdx) = dSoft;

            ySoft = waverec(cSoft,book,wname);
            ySoft = ySoft(:)';

            MSE_Soft(r) = mean((ySoft-yTrue).^2);

            % -------------------------------------------------
            % Nonnegative garrote
            % -------------------------------------------------

            dGarrote = zeros(size(d));
            active = ad > lambda;

            dGarrote(active) = ...
                d(active).*(1-lambda^2./d(active).^2);

            cGarrote = c;
            cGarrote(detailIdx) = dGarrote;

            yGarrote = waverec(cGarrote,book,wname);
            yGarrote = yGarrote(:)';

            MSE_Garrote(r) = mean((yGarrote-yTrue).^2);

            % -------------------------------------------------
            % Hard threshold
            % -------------------------------------------------

            dHard = d;
            dHard(ad <= lambda) = 0;

            cHard = c;
            cHard(detailIdx) = dHard;

            yHard = waverec(cHard,book,wname);
            yHard = yHard(:)';

            MSE_Hard(r) = mean((yHard-yTrue).^2);

            % -------------------------------------------------
            % Uniform GRIT = Beta(1,1)
            % -------------------------------------------------

            active = ad > lambda;

            for ip = 1:nP

                p = pGrid(ip);

                dGRIT = zeros(size(d));

                if p > 0 && any(active)

                    u = 1-(lambda./ad(active)).^p;
                    u = max(0,min(1,u));

                    % Uniform / Beta(1,1) CDF:
                    G = u;

                    dGRIT(active) = d(active).*G;

                end
                % For p=0, dGRIT stays exactly zero.

                cGRIT = c;
                cGRIT(detailIdx) = dGRIT;

                yGRIT = waverec(cGRIT,book,wname);
                yGRIT = yGRIT(:)';

                MSE_GRIT(r,ip) = mean((yGRIT-yTrue).^2);

            end

        end

        %% ====================================================
        % 5. AMSE
        % =====================================================

        AMSE_Noisy   = mean(MSE_Noisy);
        AMSE_Coarse  = mean(MSE_Coarse);
        AMSE_Soft    = mean(MSE_Soft);
        AMSE_Garrote = mean(MSE_Garrote);
        AMSE_Hard    = mean(MSE_Hard);

        AMSE_GRIT = mean(MSE_GRIT,1);
        MCSE_GRIT = std(MSE_GRIT,0,1)/sqrt(R);

        % -----------------------------------------------------
        % Mandatory identities
        % -----------------------------------------------------

        iP0 = find(abs(pGrid-0) < 1e-12,1);
        iP1 = find(abs(pGrid-1) < 1e-12,1);
        iP2 = find(abs(pGrid-2) < 1e-12,1);

        errP0 = max(abs(MSE_GRIT(:,iP0)-MSE_Coarse));
        errP1 = max(abs(MSE_GRIT(:,iP1)-MSE_Soft));
        errP2 = max(abs(MSE_GRIT(:,iP2)-MSE_Garrote));

        if errP0 > 1e-10
            error('Uniform GRIT p=0 does not equal coarse-only.');
        end

        if errP1 > 1e-10
            error('Uniform GRIT p=1 does not equal soft thresholding.');
        end

        if errP2 > 1e-10
            error('Uniform GRIT p=2 does not equal garrote.');
        end

        if abs(AMSE_Noisy-1) > 0.10
            warning(['Noisy AMSE = %.4f, unexpectedly far from 1 ' ...
                     'for %s, SNR=%g.'], ...
                    AMSE_Noisy,signalName,SNRvalue);
        end

        % -----------------------------------------------------
        % Oracle p on this grid
        % -----------------------------------------------------

        [BestAMSE,bestIndex] = min(AMSE_GRIT);
        BestP = pGrid(bestIndex);
        BestMCSE = MCSE_GRIT(bestIndex);

        % -----------------------------------------------------
        % Output
        % -----------------------------------------------------

        fprintf('\n---------------------------------------------\n');
        fprintf('Signal             : %s\n',signalName);
        fprintf('Wavelet            : %s\n',wname);
        fprintf('Target SNR         : %.1f\n',SNRvalue);
        fprintf('Signal E(f^2)      : %.8f\n',signalPower);
        fprintf('Achieved SNR       : %.8f\n',achievedSNR);
        fprintf('Noise variance     : %.8f\n',sigma^2);
        fprintf('Noisy AMSE         : %.8f\n',AMSE_Noisy);
        fprintf('Coarse-only AMSE   : %.8f\n',AMSE_Coarse);
        fprintf('Soft AMSE          : %.8f\n',AMSE_Soft);
        fprintf('Garrote AMSE       : %.8f\n',AMSE_Garrote);
        fprintf('Hard AMSE          : %.8f\n',AMSE_Hard);
        fprintf('Best p             : %.2f\n',BestP);
        fprintf('Best GRIT AMSE     : %.8f\n',BestAMSE);
        fprintf('Best GRIT MCSE     : %.8f\n',BestMCSE);
        fprintf('True approx energy : %.8f\n',trueApproxEnergy);
        fprintf('True detail energy : %.8f\n',trueDetailEnergy);
        fprintf('Recon max error    : %.3e\n',reconError);

        % -----------------------------------------------------
        % Store
        % -----------------------------------------------------

        fieldName = sprintf('%s_SNR%d',signalName,SNRvalue);

        Results.(fieldName).signal = signalName;
        Results.(fieldName).wavelet = wname;
        Results.(fieldName).SNR = SNRvalue;
        Results.(fieldName).signalPower = signalPower;
        Results.(fieldName).noiseVariance = sigma^2;
        Results.(fieldName).lambda = lambda;

        Results.(fieldName).pGrid = pGrid;
        Results.(fieldName).AMSE_GRIT = AMSE_GRIT;
        Results.(fieldName).MCSE_GRIT = MCSE_GRIT;

        Results.(fieldName).BestP = BestP;
        Results.(fieldName).BestAMSE = BestAMSE;
        Results.(fieldName).BestMCSE = BestMCSE;

        Results.(fieldName).AMSE_Noisy = AMSE_Noisy;
        Results.(fieldName).AMSE_Coarse = AMSE_Coarse;
        Results.(fieldName).AMSE_Soft = AMSE_Soft;
        Results.(fieldName).AMSE_Garrote = AMSE_Garrote;
        Results.(fieldName).AMSE_Hard = AMSE_Hard;

        % -----------------------------------------------------
        % Summary
        % -----------------------------------------------------

        newRow = table( ...
            string(signalName), ...
            string(wname), ...
            SNRvalue, ...
            BestP, ...
            BestAMSE, ...
            BestMCSE, ...
            AMSE_Noisy, ...
            AMSE_Coarse, ...
            AMSE_Soft, ...
            AMSE_Garrote, ...
            AMSE_Hard, ...
            'VariableNames', ...
            {'Signal','Wavelet','SNR','BestP','BestAMSE','BestMCSE', ...
             'NoisyAMSE','CoarseAMSE','SoftAMSE','GarroteAMSE','HardAMSE'});

        Summary = [Summary; newRow];

    end
end

%% ============================================================
% 6. SAVE
% =============================================================

fprintf('\n\n=============================================\n');
fprintf('FINAL SUMMARY\n');
fprintf('=============================================\n\n');

disp(Summary);

save('GRIT_Uniform_Beta11_Results.mat', ...
     'Results','Summary','pGrid','SNRlist','R','sigma','lambda');

writetable(Summary,'GRIT_Uniform_Beta11_Summary.csv');

%% ============================================================
% 7. FIGURE
%
% One panel for each SNR.
% Four signal AMSE curves in each panel.
% =============================================================

figure('Position',[100 100 1400 450]);

tiled = tiledlayout(1,nSNR, ...
    'TileSpacing','compact', ...
    'Padding','compact');

signalLabels = {'Blocks','Bumps','Doppler','HeaviSine'};
hSignal = gobjects(nSignals,1);

for iSNR = 1:nSNR

    SNRvalue = SNRlist(iSNR);

    ax = nexttile;
    hold(ax,'on');

    for iSig = 1:nSignals

        signalName = signals{iSig};
        fieldName = sprintf('%s_SNR%d',signalName,SNRvalue);

        hSignal(iSig) = plot(ax, ...
            pGrid, ...
            Results.(fieldName).AMSE_GRIT, ...
            'LineWidth',1.6);

    end

    yline(ax,1,'--','Noisy AMSE = 1', ...
        'LineWidth',1.0, ...
        'LabelHorizontalAlignment','left');

    xlabel(ax,'p');
    ylabel(ax,'AMSE');
    title(ax,sprintf('SNR = %g',SNRvalue));

    xlim(ax,[0 10]);
    grid(ax,'on');
    box(ax,'on');

    hold(ax,'off');
end

title(tiled, ...
    'Uniform GRIT / Beta(1,1): AMSE versus p', ...
    'FontWeight','bold');

lgd = legend(hSignal,signalLabels, ...
    'Orientation','horizontal', ...
    'NumColumns',4);

lgd.Layout.Tile = 'south';

exportgraphics(gcf, ...
    'GRIT_Uniform_Beta11_AllSignals.png', ...
    'Resolution',300);

fprintf('\nSaved files:\n');
fprintf('  GRIT_Uniform_Beta11_Results.mat\n');
fprintf('  GRIT_Uniform_Beta11_Summary.csv\n');
fprintf('  GRIT_Uniform_Beta11_AllSignals.png\n');


%% ============================================================
% LOCAL FUNCTION: WaveLab-style standard test signals
% =============================================================

function f = make_test_signal(name,n)

    t = (1:n)/n;

    switch name

        case 'Blocks'

            pos = [.10 .13 .15 .23 .25 .40 .44 .65 .76 .78 .81];
            hgt = [4 -5 3 -4 5 -4.2 2.1 4.3 -3.1 2.1 -4.2];

            f = zeros(1,n);

            for j = 1:numel(pos)
                f = f + hgt(j)*(1+sign(t-pos(j)))/2;
            end


        case 'Bumps'

            pos = [.10 .13 .15 .23 .25 .40 .44 .65 .76 .78 .81];

            hgt = [4 5 3 4 5 4.2 2.1 4.3 3.1 5.1 4.2];

            wid = [.005 .005 .006 .01 .01 .03 ...
                   .01 .01 .005 .008 .005];

            f = zeros(1,n);

            for j = 1:numel(pos)
                f = f + hgt(j) ./ ...
                    (1 + abs((t-pos(j))/wid(j))).^4;
            end


        case 'Doppler'

            f = sqrt(t.*(1-t)) .* ...
                sin((2*pi*1.05)./(t+.05));


        case 'HeaviSine'

            f = 4*sin(4*pi*t) ...
                - sign(t-.3) ...
                - sign(.72-t);


        otherwise

            error('Unknown signal: %s',name);

    end

end
