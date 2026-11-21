%% ============================================================
%  GRIT SIMULATION -- UNIFORM GATE = Beta(1,1)
%  From scratch
%
%  4 signals x 3 SNRs
%
%  SNR definition:
%       SNR = E(signal^2) / Var(noise)
%
%  Here Var(noise) = sigma^2 = 1, so for each requested SNR
%  the signal is rescaled to satisfy
%
%       mean(yTrue.^2) = SNR.
%
%  GRIT rule:
%
%       delta(d) = 0,                              |d| <= lambda
%       delta(d) = d * [1-(lambda/|d|)^p],        |d| >  lambda
%
%  because the Uniform gate is Beta(1,1), F(u)=u.
%
%  p grid: 0 : 0.1 : 10
%
%  p = 0  : all detail coefficients are zero
%  p = 1  : soft thresholding
%  p = 2  : nonnegative garrote
%  p -> inf: hard thresholding
%
%  Known noise sigma = 1:
%       lambda = sqrt(2*log(n))
%
%  AMSE:
%       average over Monte Carlo replicates of
%       mean((estimate - true_signal).^2)
% ============================================================

clear;
clc;
close all;

rng(123);

% Folder containing MakeSignal, MakeONFilter, dwtr, idwtr
addpath('C:/Users/vijin/OneDrive/Desktop/Wavelets/Project_7_GRIT');

%% ============================================================
% 1. SETTINGS
% =============================================================

n = 1024;
L = 5;
R = 1000;

SNRlist = [3 5 7];

signals = {'Blocks','Bumps','Doppler','HeaviSine'};
nSignals = numel(signals);
nSNR = numel(SNRlist);

pGrid = 0:0.1:10;
nP = numel(pGrid);

sigma = 1;
lambda = sigma*sqrt(2*log(n));

Results = struct();
Summary = table();

fprintf('\n=============================================\n');
fprintf('UNIFORM GRIT = Beta(1,1)\n');
fprintf('n = %d, L = %d, R = %d\n',n,L,R);
fprintf('sigma = %.6f, lambda = %.6f\n',sigma,lambda);
fprintf('p grid = 0:0.1:10\n');
fprintf('=============================================\n');

%% ============================================================
% 2. LOOP OVER SIGNALS
% =============================================================

for iSig = 1:nSignals

    signalName = signals{iSig};

    % ---------------------------------------------------------
    % Signal-specific wavelet
    % ---------------------------------------------------------

    switch signalName
        case 'Blocks'
            wtype = 'Haar';
            filterSize = 2;

        case 'Bumps'
            wtype = 'Daubechies';
            filterSize = 6;

        case {'Doppler','HeaviSine'}
            wtype = 'Symmlet';
            filterSize = 8;

        otherwise
            error('Unknown signal: %s',signalName);
    end

    filt = MakeONFilter(wtype,filterSize);

    % ---------------------------------------------------------
    % Original WaveLab test signal
    % ---------------------------------------------------------

    yBase = MakeSignal(signalName,n);
    yBase = yBase(:)';

    basePower = mean(yBase.^2);

    if basePower <= 0
        error('Nonpositive signal power for %s.',signalName);
    end

    % ---------------------------------------------------------
    % DWT coefficient layout for supplied dwtr/idwtr:
    %
    % [ C_L | D_L | D_{L-1} | ... | D_1 ]
    %
    % Number of coarse scaling coefficients = n/2^L
    % ---------------------------------------------------------

    nScaling = n/(2^L);

    scalingIdx = 1:nScaling;
    detailIdx  = (nScaling+1):n;

    %% ========================================================
    % 3. LOOP OVER SNR
    % =========================================================

    for iSNR = 1:nSNR

        SNRvalue = SNRlist(iSNR);

        % -----------------------------------------------------
        % RESCALE SIGNAL -- THIS DEFINES THE REQUESTED SNR
        %
        % Var(noise)=1, so force mean(yTrue.^2)=SNRvalue.
        % -----------------------------------------------------

        scaleFactor = sqrt(SNRvalue/basePower);
        yTrue = scaleFactor*yBase;

        signalPower = mean(yTrue.^2);
        achievedSNR = signalPower/(sigma^2);

        if abs(achievedSNR-SNRvalue) > 1e-12
            error('SNR scaling failed for %s at SNR=%g.', ...
                  signalName,SNRvalue);
        end

        % -----------------------------------------------------
        % DWT / IDWT sanity checks using the TRUE rescaled signal
        % -----------------------------------------------------

        wtTrue = dwtr(yTrue,L,filt);
        yRecon = idwtr(wtTrue,L,filt);
        yRecon = yRecon(:)';

        reconError = max(abs(yRecon-yTrue));

        if reconError > 1e-8
            error(['DWT/IDWT reconstruction failed for %s, SNR=%g. ' ...
                   'Max error = %.3e'], ...
                  signalName,SNRvalue,reconError);
        end

        timeEnergy = sum(yTrue.^2);
        waveEnergy = sum(wtTrue.^2);
        parsevalRelErr = abs(waveEnergy-timeEnergy)/max(timeEnergy,eps);

        if parsevalRelErr > 1e-8
            warning(['Parseval check for %s, SNR=%g: relative error %.3e. ' ...
                     'Filter/transform may not be exactly orthonormal.'], ...
                    signalName,SNRvalue,parsevalRelErr);
        end

        % True signal energy split across coarse/details
        trueCoarseEnergy = sum(wtTrue(scalingIdx).^2)/n;
        trueDetailEnergy = sum(wtTrue(detailIdx).^2)/n;

        % -----------------------------------------------------
        % Monte Carlo storage
        % -----------------------------------------------------

        MSE_Noisy  = zeros(R,1);
        MSE_Coarse = zeros(R,1);
        MSE_Soft   = zeros(R,1);
        MSE_Garrote = zeros(R,1);
        MSE_Hard   = zeros(R,1);

        MSE_GRIT = zeros(R,nP);

        %% ====================================================
        % 4. MONTE CARLO LOOP
        % =====================================================

        for r = 1:R

            % Unit-variance Gaussian noise
            noise = randn(1,n);
            yNoisy = yTrue + noise;

            % Raw noisy MSE
            MSE_Noisy(r) = mean((yNoisy-yTrue).^2);

            % DWT
            wt = dwtr(yNoisy,L,filt);

            d = wt(detailIdx);

            % -------------------------------------------------
            % Coarse-only reference
            % -------------------------------------------------

            wtCoarse = wt;
            wtCoarse(detailIdx) = 0;

            yCoarse = idwtr(wtCoarse,L,filt);
            yCoarse = yCoarse(:)';

            MSE_Coarse(r) = mean((yCoarse-yTrue).^2);

            % -------------------------------------------------
            % Soft reference: must equal Uniform GRIT at p=1
            % -------------------------------------------------

            dSoft = sign(d).*max(abs(d)-lambda,0);

            wtSoft = wt;
            wtSoft(detailIdx) = dSoft;

            ySoft = idwtr(wtSoft,L,filt);
            ySoft = ySoft(:)';

            MSE_Soft(r) = mean((ySoft-yTrue).^2);

            % -------------------------------------------------
            % Garrote reference: must equal Uniform GRIT at p=2
            % -------------------------------------------------

            dGarrote = zeros(size(d));
            active = abs(d) > lambda;

            dGarrote(active) = ...
                d(active).*(1-lambda^2./d(active).^2);

            wtGarrote = wt;
            wtGarrote(detailIdx) = dGarrote;

            yGarrote = idwtr(wtGarrote,L,filt);
            yGarrote = yGarrote(:)';

            MSE_Garrote(r) = mean((yGarrote-yTrue).^2);

            % -------------------------------------------------
            % Hard reference: p -> infinity limit
            % -------------------------------------------------

            dHard = d;
            dHard(abs(dHard) <= lambda) = 0;

            wtHard = wt;
            wtHard(detailIdx) = dHard;

            yHard = idwtr(wtHard,L,filt);
            yHard = yHard(:)';

            MSE_Hard(r) = mean((yHard-yTrue).^2);

            % -------------------------------------------------
            % Uniform GRIT = Beta(1,1)
            % -------------------------------------------------

            absd = abs(d);
            active = absd > lambda;

            for ip = 1:nP

                p = pGrid(ip);

                dGRIT = zeros(size(d));

                if any(active)

                    da = d(active);
                    ada = absd(active);

                    if p == 0
                        % u = 1-(lambda/|d|)^0 = 0
                        % Uniform gate F(u)=u => all details zero.
                        G = zeros(size(da));
                    else
                        u = 1-(lambda./ada).^p;
                        u = max(0,min(1,u));

                        % Beta(1,1) = Uniform CDF
                        G = u;
                    end

                    dGRIT(active) = da.*G;
                end

                wtGRIT = wt;
                wtGRIT(detailIdx) = dGRIT;

                yGRIT = idwtr(wtGRIT,L,filt);
                yGRIT = yGRIT(:)';

                MSE_GRIT(r,ip) = mean((yGRIT-yTrue).^2);

            end

        end

        %% ====================================================
        % 5. MONTE CARLO AVERAGES
        % =====================================================

        AMSE_Noisy   = mean(MSE_Noisy);
        AMSE_Coarse  = mean(MSE_Coarse);
        AMSE_Soft    = mean(MSE_Soft);
        AMSE_Garrote = mean(MSE_Garrote);
        AMSE_Hard    = mean(MSE_Hard);

        AMSE_GRIT = mean(MSE_GRIT,1);
        MCSE_GRIT = std(MSE_GRIT,0,1)/sqrt(R);

        % -----------------------------------------------------
        % Exact/near-exact identity checks
        % -----------------------------------------------------

        iP0 = find(abs(pGrid-0) < 1e-12,1);
        iP1 = find(abs(pGrid-1) < 1e-12,1);
        iP2 = find(abs(pGrid-2) < 1e-12,1);

        errP0 = max(abs(MSE_GRIT(:,iP0)-MSE_Coarse));
        errP1 = max(abs(MSE_GRIT(:,iP1)-MSE_Soft));
        errP2 = max(abs(MSE_GRIT(:,iP2)-MSE_Garrote));

        if errP0 > 1e-10
            error('p=0 Uniform GRIT does not match coarse-only.');
        end

        if errP1 > 1e-10
            error('p=1 Uniform GRIT does not match soft thresholding.');
        end

        if errP2 > 1e-10
            error('p=2 Uniform GRIT does not match garrote.');
        end

        if abs(AMSE_Noisy-1) > 0.10
            warning(['Noisy AMSE %.4f is unexpectedly far from 1 ' ...
                     'for %s, SNR=%g.'], ...
                    AMSE_Noisy,signalName,SNRvalue);
        end

        % -----------------------------------------------------
        % Best p
        % -----------------------------------------------------

        [BestAMSE,bestIndex] = min(AMSE_GRIT);
        BestP = pGrid(bestIndex);
        BestMCSE = MCSE_GRIT(bestIndex);

        % -----------------------------------------------------
        % Print diagnostics
        % -----------------------------------------------------

        fprintf('\n---------------------------------------------\n');
        fprintf('Signal: %-10s  SNR: %g  Wavelet: %s %d\n', ...
            signalName,SNRvalue,wtype,filterSize);
        fprintf('Signal power        = %.8f\n',signalPower);
        fprintf('Achieved SNR        = %.8f\n',achievedSNR);
        fprintf('Noisy AMSE          = %.8f\n',AMSE_Noisy);
        fprintf('Coarse-only AMSE    = %.8f\n',AMSE_Coarse);
        fprintf('Soft AMSE           = %.8f\n',AMSE_Soft);
        fprintf('Garrote AMSE        = %.8f\n',AMSE_Garrote);
        fprintf('Hard AMSE           = %.8f\n',AMSE_Hard);
        fprintf('Best Uniform GRIT p = %.2f\n',BestP);
        fprintf('Best Uniform AMSE   = %.8f\n',BestAMSE);
        fprintf('Best Uniform MCSE   = %.8f\n',BestMCSE);
        fprintf('True coarse energy  = %.8f\n',trueCoarseEnergy);
        fprintf('True detail energy  = %.8f\n',trueDetailEnergy);
        fprintf('Recon max error     = %.3e\n',reconError);
        fprintf('Parseval rel. error = %.3e\n',parsevalRelErr);

        % -----------------------------------------------------
        % Store
        % -----------------------------------------------------

        fieldName = sprintf('%s_SNR%d',signalName,SNRvalue);

        Results.(fieldName).signal = signalName;
        Results.(fieldName).SNR = SNRvalue;
        Results.(fieldName).wavelet = wtype;
        Results.(fieldName).filterSize = filterSize;

        Results.(fieldName).signalPower = signalPower;
        Results.(fieldName).sigma = sigma;
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

        Results.(fieldName).trueCoarseEnergy = trueCoarseEnergy;
        Results.(fieldName).trueDetailEnergy = trueDetailEnergy;
        Results.(fieldName).reconError = reconError;
        Results.(fieldName).parsevalRelErr = parsevalRelErr;

        % -----------------------------------------------------
        % Summary table
        % -----------------------------------------------------

        newRow = table( ...
            string(signalName), ...
            SNRvalue, ...
            string(wtype), ...
            filterSize, ...
            BestP, ...
            BestAMSE, ...
            BestMCSE, ...
            AMSE_Noisy, ...
            AMSE_Coarse, ...
            AMSE_Soft, ...
            AMSE_Garrote, ...
            AMSE_Hard, ...
            trueDetailEnergy, ...
            'VariableNames', ...
            {'Signal','SNR','Wavelet','FilterSize', ...
             'BestP','BestAMSE','BestMCSE', ...
             'NoisyAMSE','CoarseAMSE','SoftAMSE', ...
             'GarroteAMSE','HardAMSE','TrueDetailEnergy'});

        Summary = [Summary; newRow];

    end
end

%% ============================================================
% 6. DISPLAY AND SAVE
% =============================================================

fprintf('\n\n=============================================\n');
fprintf('FINAL SUMMARY -- UNIFORM GRIT Beta(1,1)\n');
fprintf('=============================================\n\n');

disp(Summary);

save('GRIT_Uniform_Beta11_Results.mat', ...
     'Results','Summary','pGrid','SNRlist','R','sigma','lambda');

writetable(Summary,'GRIT_Uniform_Beta11_Summary.csv');

%% ============================================================
% 7. FIGURE
%
% Three panels: SNR = 3, 5, 7.
% In each panel, show the four signals' AMSE curves versus p.
% Horizontal line at AMSE = 1 is the raw-noise baseline.
% =============================================================

figure('Position',[100 100 1400 450]);

t = tiledlayout(1,nSNR, ...
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

        AMSEcurve = Results.(fieldName).AMSE_GRIT;

        hSignal(iSig) = plot(ax, ...
            pGrid,AMSEcurve, ...
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

title(t, ...
    'Uniform GRIT / Beta(1,1): AMSE versus p', ...
    'FontWeight','bold');

lgd = legend(hSignal,signalLabels, ...
    'Orientation','horizontal', ...
    'NumColumns',4);

lgd.Layout.Tile = 'south';

exportgraphics(gcf, ...
    'GRIT_Uniform_Beta11_AllSignals.png', ...
    'Resolution',300);

fprintf('\nSaved:\n');
fprintf('1. GRIT_Uniform_Beta11_Results.mat\n');
fprintf('2. GRIT_Uniform_Beta11_Summary.csv\n');
fprintf('3. GRIT_Uniform_Beta11_AllSignals.png\n');
