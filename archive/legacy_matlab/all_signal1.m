%% ============================================================
%  GRIT SIMULATION A
%  ============================================================
%
%  4 Signals x 3 SNRs x 5 GRIT Gates
%
%  Signals and wavelets:
%     Blocks      -> Haar
%     Bumps       -> Daubechies 6
%     Doppler     -> Symmlet 8
%     HeaviSine   -> Symmlet 8
%
%  Decomposition level:
%     L = 5
%
%  SNR:
%     3, 5, 7
%
%  GRIT Gates:
%     1. Uniform
%     2. Topp-Leone(alpha = 1)
%     3. Beta(2,2)
%     4. Kumaraswamy(2,2)
%     5. Raised-Cosine
%
%  Reference methods:
%     Noisy
%     Coarse-only
%     Hard
%     Soft
%     Nonnegative Garrote
%
%  Threshold:
%     lambda = sigmaHat * sqrt(2*log(n))
%
%  Noise estimate:
%     MAD from finest-level wavelet coefficients
%
% ============================================================

clear;
clc;
close all;

rng(123);      
addpath('C:/Users/vijin/OneDrive/Desktop/Wavelets/Project_7_GRIT')


%% ============================================================
% 1. BASIC SETTINGS
% =============================================================

n = 1024;
L = 5;

SNRlist = [3 5 7];

R = 1000;

signals = {'Blocks','Bumps','Doppler','HeaviSine'};

gateNames = {'Uniform', ...
             'ToppLeone', ...
             'Beta22', ...
             'Kumaraswamy22', ...
             'RaisedCosine'};

nSignals = length(signals);
nSNR     = length(SNRlist);
nGates   = length(gateNames);


%% ============================================================
% 2. p GRID
% =============================================================

pGrid = [0.01 0.05 0.10 0.25 0.50 0.75 1 ...
         1.5 2];

nP = length(pGrid);


%% ============================================================
% 3. STORAGE
% =============================================================

Summary = table();

Results = struct();


%% ============================================================
% 4. MAIN LOOP OVER SIGNALS
% =============================================================

for iSig = 1:nSignals

    signalName = signals{iSig};


    %% --------------------------------------------------------
    % Select signal-specific wavelet
    % ---------------------------------------------------------

    switch signalName

        case 'Blocks'

            wtype = 'Haar';
            filtersize = 2;

        case 'Bumps'

            wtype = 'Daubechies';
            filtersize = 6;

        case {'Doppler','HeaviSine'}

            wtype = 'Symmlet';
            filtersize = 8;

        otherwise

            error('Unknown signal: %s',signalName);

    end


    filt = MakeONFilter(wtype,filtersize);


    %% --------------------------------------------------------
    % Generate true signal
    % ---------------------------------------------------------

    yTrue = MakeSignal(signalName,n);

    yTrue = yTrue(:)';

    %signalSD = std(yTrue);
    signalPower = mean(yTrue.^2);


    %% --------------------------------------------------------
    % Wavelet coefficient indexing
    % ---------------------------------------------------------

    nScaling = n/(2^L);

    scalingIdx = 1:nScaling;
    detailIdx  = (nScaling+1):n;

    finestIdx = (n/2+1):n;


    %% ========================================================
    % 5. LOOP OVER SNR
    % =========================================================

    for iSNR = 1:nSNR

        SNRvalue = SNRlist(iSNR);


        fprintf('\n');
        fprintf('=============================================\n');
        fprintf('Signal  : %s\n',signalName);
        fprintf('SNR     : %g\n',SNRvalue);
        fprintf('Wavelet : %s %d\n',wtype,filtersize);
        fprintf('=============================================\n');


        %% ----------------------------------------------------
        % True noise standard deviation
        % -----------------------------------------------------

        sigma = sqrt(signalPower/SNRvalue);


        %% ----------------------------------------------------
        % Monte Carlo storage
        % -----------------------------------------------------

        MSE_Noisy   = zeros(R,1);
        MSE_Coarse  = zeros(R,1);
        MSE_Hard    = zeros(R,1);
        MSE_Soft    = zeros(R,1);
        MSE_Garrote = zeros(R,1);

        sigmaHatSave = zeros(R,1);
        lambdaSave   = zeros(R,1);


        % MSE_GRIT dimensions:
        %
        % R x number of p values x number of gates

        MSE_GRIT = zeros(R,nP,nGates);


        %% ====================================================
        % 6. MONTE CARLO LOOP
        % =====================================================

        for r = 1:R


            %% ------------------------------------------------
            % Generate Gaussian noise
            % -------------------------------------------------

            noise = sigma*randn(1,n);

            yNoisy = yTrue + noise;


            %% ------------------------------------------------
            % MSE of noisy signal
            % -------------------------------------------------

            MSE_Noisy(r) = ...
                mean((yNoisy-yTrue).^2);


            %% ------------------------------------------------
            % Discrete wavelet transform
            % -------------------------------------------------

            wtdata = dwtr(yNoisy,L,filt);


            %% ------------------------------------------------
            % Estimate sigma using MAD
            % -------------------------------------------------

            finest = wtdata(finestIdx);

            sigmaHat = ...
                median(abs(finest-median(finest)))/0.6745;


            %% ------------------------------------------------
            % Universal threshold
            % -------------------------------------------------

            lambda = sigmaHat*sqrt(2*log(n));

            sigmaHatSave(r) = sigmaHat;
            lambdaSave(r)   = lambda;


            %% =================================================
            % 6A. COARSE-ONLY
            % ==================================================

            wtCoarse = wtdata;

            % Set ALL detail coefficients to zero
            wtCoarse(detailIdx) = 0;

            yCoarse = idwtr(wtCoarse,L,filt);

            yCoarse = yCoarse(:)';

            MSE_Coarse(r) = ...
                mean((yCoarse-yTrue).^2);


            %% =================================================
            % 6B. HARD THRESHOLDING
            % ==================================================

            wtHard = wtdata;

            d = wtHard(detailIdx);

            d(abs(d) <= lambda) = 0;

            wtHard(detailIdx) = d;

            yHard = idwtr(wtHard,L,filt);

            yHard = yHard(:)';

            MSE_Hard(r) = ...
                mean((yHard-yTrue).^2);


            %% =================================================
            % 6C. SOFT THRESHOLDING
            % ==================================================

            wtSoft = wtdata;

            d = wtSoft(detailIdx);

            dSoft = ...
                sign(d).*max(abs(d)-lambda,0);

            wtSoft(detailIdx) = dSoft;

            ySoft = idwtr(wtSoft,L,filt);

            ySoft = ySoft(:)';

            MSE_Soft(r) = ...
                mean((ySoft-yTrue).^2);


            %% =================================================
            % 6D. NONNEGATIVE GARROTE
            % ==================================================

            wtGarrote = wtdata;

            d = wtGarrote(detailIdx);

            dGarrote = zeros(size(d));

            active = abs(d) > lambda;

            dGarrote(active) = ...
                d(active).* ...
                (1-lambda^2./d(active).^2);

            wtGarrote(detailIdx) = dGarrote;

            yGarrote = idwtr(wtGarrote,L,filt);

            yGarrote = yGarrote(:)';

            MSE_Garrote(r) = ...
                mean((yGarrote-yTrue).^2);


            %% =================================================
            % 6E. GRIT
            % ==================================================

            originalDetails = wtdata(detailIdx);


            for ip = 1:nP

                p = pGrid(ip);


                for ig = 1:nGates

                    gateName = gateNames{ig};


                    %% -----------------------------------------
                    % Apply GRIT threshold
                    % ------------------------------------------

                    dGRIT = GRITThresholdMaster( ...
                        originalDetails, ...
                        lambda, ...
                        p, ...
                        gateName);


                    %% -----------------------------------------
                    % Replace detail coefficients
                    % ------------------------------------------

                    wtGRIT = wtdata;

                    wtGRIT(detailIdx) = dGRIT;


                    %% -----------------------------------------
                    % Inverse wavelet transform
                    % ------------------------------------------

                    yGRIT = idwtr(wtGRIT,L,filt);

                    yGRIT = yGRIT(:)';


                    %% -----------------------------------------
                    % Signal-domain MSE
                    % ------------------------------------------

                    MSE_GRIT(r,ip,ig) = ...
                        mean((yGRIT-yTrue).^2);

                end

            end

        end


        %% ====================================================
        % 7. MONTE CARLO SUMMARY
        % =====================================================


        %% Noisy

        AMSE_Noisy = mean(MSE_Noisy);

        MCSE_Noisy = ...
            std(MSE_Noisy)/sqrt(R);


        %% Coarse-only

        AMSE_Coarse = mean(MSE_Coarse);

        MCSE_Coarse = ...
            std(MSE_Coarse)/sqrt(R);


        %% Hard

        AMSE_Hard = mean(MSE_Hard);

        MCSE_Hard = ...
            std(MSE_Hard)/sqrt(R);


        %% Soft

        AMSE_Soft = mean(MSE_Soft);

        MCSE_Soft = ...
            std(MSE_Soft)/sqrt(R);


        %% Garrote

        AMSE_Garrote = mean(MSE_Garrote);

        MCSE_Garrote = ...
            std(MSE_Garrote)/sqrt(R);


        %% ----------------------------------------------------
        % GRIT AMSE and MCSE
        % -----------------------------------------------------

        AMSE_GRIT = zeros(nP,nGates);
        MCSE_GRIT = zeros(nP,nGates);


        for ig = 1:nGates

            for ip = 1:nP

                tempMSE = MSE_GRIT(:,ip,ig);

                AMSE_GRIT(ip,ig) = ...
                    mean(tempMSE);

                MCSE_GRIT(ip,ig) = ...
                    std(tempMSE)/sqrt(R);

            end

        end


        %% ====================================================
        % 8. RESULTS
        % =====================================================

        fprintf('\nTrue sigma       = %.8f\n',sigma);

        fprintf('Mean sigmaHat    = %.8f\n', ...
            mean(sigmaHatSave));

        fprintf('Mean lambda      = %.8f\n\n', ...
            mean(lambdaSave));


        fprintf(['Noisy AMSE       = %.8f   ' ...
                 'MCSE = %.8f\n'], ...
            AMSE_Noisy,MCSE_Noisy);


        fprintf(['Coarse-only AMSE = %.8f   ' ...
                 'MCSE = %.8f\n'], ...
            AMSE_Coarse,MCSE_Coarse);


        fprintf(['Hard AMSE        = %.8f   ' ...
                 'MCSE = %.8f\n'], ...
            AMSE_Hard,MCSE_Hard);


        fprintf(['Soft AMSE        = %.8f   ' ...
                 'MCSE = %.8f\n'], ...
            AMSE_Soft,MCSE_Soft);


        fprintf(['Garrote AMSE     = %.8f   ' ...
                 'MCSE = %.8f\n\n'], ...
            AMSE_Garrote,MCSE_Garrote);


        %% ====================================================
        % 9. FIND ORACLE p FOR EACH GRIT GATE
        % =====================================================

        BestP    = zeros(nGates,1);
        BestAMSE = zeros(nGates,1);
        BestMCSE = zeros(nGates,1);

        Boundary = strings(nGates,1);


        for ig = 1:nGates


            [BestAMSE(ig),bestIndex] = ...
                min(AMSE_GRIT(:,ig));


            BestP(ig) = ...
                pGrid(bestIndex);


            BestMCSE(ig) = ...
                MCSE_GRIT(bestIndex,ig);


            %% ---------------------------------------------
            % Check whether minimum occurs at boundary
            % ----------------------------------------------

            if bestIndex == 1

                Boundary(ig) = "Lower";

            elseif bestIndex == nP

                Boundary(ig) = "Upper";

            else

                Boundary(ig) = "No";

            end


            fprintf('%-18s ',gateNames{ig});

            fprintf('Best p = %6.2f   ', ...
                BestP(ig));

            fprintf('AMSE = %.8f   ', ...
                BestAMSE(ig));

            fprintf('MCSE = %.8f   ', ...
                BestMCSE(ig));

            fprintf('Boundary = %s\n', ...
                Boundary(ig));

        end


        %% ====================================================
        % 10. STORE DETAILED RESULTS
        % =====================================================

        fieldName = ...
            sprintf('%s_SNR%d', ...
            signalName,SNRvalue);


        Results.(fieldName).signal = ...
            signalName;

        Results.(fieldName).SNR = ...
            SNRvalue;

        Results.(fieldName).wavelet = ...
            wtype;

        Results.(fieldName).filterSize = ...
            filtersize;

        Results.(fieldName).sigma = ...
            sigma;

        Results.(fieldName).meanSigmaHat = ...
            mean(sigmaHatSave);

        Results.(fieldName).meanLambda = ...
            mean(lambdaSave);

        Results.(fieldName).pGrid = ...
            pGrid;

        Results.(fieldName).gateNames = ...
            gateNames;

        Results.(fieldName).AMSE_GRIT = ...
            AMSE_GRIT;

        Results.(fieldName).MCSE_GRIT = ...
            MCSE_GRIT;

        Results.(fieldName).BestP = ...
            BestP;

        Results.(fieldName).BestAMSE = ...
            BestAMSE;

        Results.(fieldName).BestMCSE = ...
            BestMCSE;

        Results.(fieldName).Boundary = ...
            Boundary;


        %% Reference methods

        Results.(fieldName).AMSE_Noisy = ...
            AMSE_Noisy;

        Results.(fieldName).MCSE_Noisy = ...
            MCSE_Noisy;

        Results.(fieldName).AMSE_Coarse = ...
            AMSE_Coarse;

        Results.(fieldName).MCSE_Coarse = ...
            MCSE_Coarse;

        Results.(fieldName).AMSE_Hard = ...
            AMSE_Hard;

        Results.(fieldName).MCSE_Hard = ...
            MCSE_Hard;

        Results.(fieldName).AMSE_Soft = ...
            AMSE_Soft;

        Results.(fieldName).MCSE_Soft = ...
            MCSE_Soft;

        Results.(fieldName).AMSE_Garrote = ...
            AMSE_Garrote;

        Results.(fieldName).MCSE_Garrote = ...
            MCSE_Garrote;


        %% ====================================================
        % 11. ADD TO SUMMARY TABLE
        % =====================================================

        for ig = 1:nGates


            newRow = table( ...
                string(signalName), ...
                SNRvalue, ...
                string(wtype), ...
                filtersize, ...
                string(gateNames{ig}), ...
                BestP(ig), ...
                BestAMSE(ig), ...
                BestMCSE(ig), ...
                Boundary(ig), ...
                AMSE_Coarse, ...
                AMSE_Hard, ...
                AMSE_Soft, ...
                AMSE_Garrote, ...
                'VariableNames', ...
                {'Signal', ...
                 'SNR', ...
                 'Wavelet', ...
                 'FilterSize', ...
                 'Gate', ...
                 'BestP', ...
                 'BestAMSE', ...
                 'BestMCSE', ...
                 'Boundary', ...
                 'CoarseAMSE', ...
                 'HardAMSE', ...
                 'SoftAMSE', ...
                 'GarroteAMSE'});


            Summary = [Summary; newRow];

        end


    end

end


%% ============================================================
% 12. DISPLAY FINAL SUMMARY TABLE
% =============================================================

fprintf('\n\n');
fprintf('=============================================\n');
fprintf('FINAL GRIT SIMULATION SUMMARY\n');
fprintf('=============================================\n\n');

disp(Summary);


%% ============================================================
% 13. SAVE NUMERICAL RESULTS
% =============================================================

save('GRIT_SimulationA_Results.mat', ...
     'Results', ...
     'Summary', ...
     'pGrid', ...
     'SNRlist', ...
     'R');


writetable(Summary, ...
    'GRIT_SimulationA_Summary.csv');


%% ============================================================
% 14. CREATE FIGURES BY GRIT GATE / DENSITY
%
% One figure for each GRIT gate.
% Each figure contains three panels, one for each SNR.
% Within each panel, the four signal AMSE curves are shown
% as functions of p:
%
%     Blocks, Bumps, Doppler, HeaviSine
%
% Thus the display is the transpose of the original plotting
% organization: gate/density is fixed and signals are compared.
% =============================================================

signalLabels = {'Blocks','Bumps','Doppler','HeaviSine'};

for ig = 1:nGates

    gateName = gateNames{ig};

    %% --------------------------------------------------------
    % New figure for this gate / density
    % ---------------------------------------------------------

    figure('Position',[100 100 1350 430]);

    t = tiledlayout(1,nSNR, ...
        'TileSpacing','compact', ...
        'Padding','compact');

    hSignal = gobjects(nSignals,1);

    %% ========================================================
    % Loop over SNR values
    % =========================================================

    for iSNR = 1:nSNR

        SNRvalue = SNRlist(iSNR);

        ax = nexttile;
        hold(ax,'on');

        %% ----------------------------------------------------
        % Plot AMSE versus p for all four signals
        % -----------------------------------------------------

        for iSig = 1:nSignals

            signalName = signals{iSig};

            fieldName = ...
                sprintf('%s_SNR%d', ...
                signalName,SNRvalue);

            AMSE_GRIT = ...
                Results.(fieldName).AMSE_GRIT;

            hSignal(iSig) = ...
                plot(ax, ...
                pGrid, ...
                AMSE_GRIT(:,ig), ...
                '-o', ...
                'LineWidth',1.5, ...
                'MarkerSize',4);

        end

        %% ----------------------------------------------------
        % Labels
        % -----------------------------------------------------

        xlabel(ax,'p');
        ylabel(ax,'AMSE');

        title(ax, ...
            sprintf('SNR = %g',SNRvalue));

        grid(ax,'on');
        box(ax,'on');

        hold(ax,'off');

    end

    %% --------------------------------------------------------
    % Overall title and signal legend
    % -----------------------------------------------------

    switch gateName
        case 'Uniform'
            gateLabel = 'Uniform';
        case 'ToppLeone'
            gateLabel = 'Topp-Leone (\alpha = 1)';
        case 'Beta22'
            gateLabel = 'Beta(2,2)';
        case 'Kumaraswamy22'
            gateLabel = 'Kumaraswamy(2,2)';
        case 'RaisedCosine'
            gateLabel = 'Raised-Cosine';
        otherwise
            gateLabel = gateName;
    end

    title(t, ...
        sprintf('GRIT Wavelet Denoising: %s Gate', ...
        gateLabel), ...
        'FontWeight','bold');

    lgd = legend(hSignal, ...
        signalLabels, ...
        'Orientation','horizontal', ...
        'NumColumns',4);

    lgd.Layout.Tile = 'south';

    %% --------------------------------------------------------
    % Save figure
    % -----------------------------------------------------

    figureName = ...
        sprintf('GRIT_ByGate_%s.png',gateName);

    exportgraphics(gcf, ...
        figureName, ...
        'Resolution',300);

end


%% ============================================================
% =============================================================

fprintf('\n=============================================\n');
fprintf('SIMULATION COMPLETED\n');
fprintf('=============================================\n');

fprintf('\nFiles saved:\n');

fprintf('1. GRIT_SimulationA_Results.mat\n');
fprintf('2. GRIT_SimulationA_Summary.csv\n');

for ig = 1:nGates

    fprintf('%d. GRIT_ByGate_%s.png\n', ...
        ig+2,gateNames{ig});

end

function delta = ...
    GRITThresholdMaster(x,lambda,p,gateName)


    %% --------------------------------------------------------
    % Initialize output
    % ---------------------------------------------------------

    delta = zeros(size(x));


    %% --------------------------------------------------------
    % Magnitudes
    % ---------------------------------------------------------

    absx = abs(x);


    %% --------------------------------------------------------
    % Active coefficients
    %
    % GRIT = 0 when |x| <= lambda
    % ---------------------------------------------------------

    active = absx > lambda;


    if ~any(active)
        return;
    end


    xa = x(active);

    ax = absx(active);


    %% --------------------------------------------------------
    % Reciprocal coordinate
    %
    % u = 1 - (lambda/|x|)^p
    % ---------------------------------------------------------

    u = 1-(lambda./ax).^p;


    %% --------------------------------------------------------
    % Numerical protection
    %
    % Theoretically:
    %       0 < u < 1
    % ---------------------------------------------------------

    u = max(0,min(1,u));


    %% ========================================================
    % CDF GATE
    % =========================================================

    switch gateName


        %% ----------------------------------------------------
        % 1. UNIFORM
        %
        % G(u) = u
        % -----------------------------------------------------

        case 'Uniform'

            G = u;


        %% ----------------------------------------------------
        % 2. TOPP-LEONE
        %
        % alpha = 1
        %
        % G(u) = [u(2-u)]^alpha
        % -----------------------------------------------------

        case 'ToppLeone'

            alpha = 1;

            G = (u.*(2-u)).^alpha;


        %% ----------------------------------------------------
        % 3. BETA(2,2)
        %
        % G(u) = I_u(2,2)
        % -----------------------------------------------------

        case 'Beta22'

            G = betainc(u,2,2);


        %% ----------------------------------------------------
        % 4. KUMARASWAMY(2,2)
        %
        % G(u) = 1-(1-u^a)^b
        %
        % a = 2
        % b = 2
        % -----------------------------------------------------

        case 'Kumaraswamy22'

            a = 2;
            b = 2;

            G = 1-(1-u.^a).^b;


        %% ----------------------------------------------------
        % 5. RAISED-COSINE
        %
        % G(u) =
        % u - sin(2*pi*u)/(2*pi)
        % -----------------------------------------------------

        case 'RaisedCosine'

            G = ...
                u-sin(2*pi*u)/(2*pi);


        otherwise

            error('Unknown gate: %s', ...
                gateName);

    end


    %% --------------------------------------------------------
    % Numerical protection for CDF
    % ---------------------------------------------------------

    G = max(0,min(1,G));


    %% --------------------------------------------------------
    % GRIT estimator
    %
    % delta(x) = x G(u)
    % ---------------------------------------------------------

    delta(active) = xa.*G;

end