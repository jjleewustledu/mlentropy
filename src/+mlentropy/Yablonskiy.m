classdef Yablonskiy < mlentropy.Bayesian
	%% YABLONSKIY
    % 
	%  Version $Revision$ was created $Date$ by $Author$  
 	%  and checked into svn repository $URL$ 
 	%  Developed on Matlab 7.10.0.499 (R2010a) 
 	%  $Id$ 
 	%  N.B. classdef (Sealed, Hidden, InferiorClasses = {?class1,?class2}, ConstructOnLoad) 

    properties (Constant)
        GYROMAG = 4252.94; % Hz/G
        B0      = 30000;   % G
        LABELS  = {'Delta_chi0','Hct','R2t','Y','zeta'};
    end
    
	properties 
 		% N.B. (Abstract, Access='private', GetAccess='protected', SetAccess='protected', ... 
 		%       Constant, Dependent, Hidden, Transient) 
        observed    %         by experiment
        designMat   % n x k   numeric arrays
        priors      % cell of function handles
        posterior   %         function handle
        dposterior
        model       %         function handle
        dmodel
        residual
        dresidual
        nsamples = 2000;
        burnInTime = 0;
        thinningMultiple = 1;
        intercept = struct('mu',0,'sigma',20);
        slope = struct('mu',0,'sigma',20);
        Delta_chi0 % NIfTI
        Hct        % scalar or NIfTI
        Y          % NIfTI
        R2t        % NIfTI, R2 of solid tissue
        zeta       % NIfTI, DBV
        initial
    end 

    properties (SetAccess='private')
        designMat_internal
        trace_internal
        modelName   = 'Yablonskiy'
        modelParams
        modelParamsMin
        modelParamsMax
        modelParamsIndex
    end

	methods 

 		function this = Yablonskiy(observed) 
 			%% LOGISTIC (ctor) 
 			%  Usage:  obj = Yablonskiy(observed)
			%                           ^ struct w/ fields signal, times
            this = this@mlentropy.Bayesian;
            import mlfourd.* mlentropy.*;
            [this.modelParams,this.modelParamsMin,this.modelParamsMax,this.modelParamsIndex] = Yablonskiy.modelParams0;
%                this.initial     = zeros(1,length(this.LABELS));
%             for in = 1:length(this.initial)
%                 this.initial(in) = this.modelParams(this.LABELS{in}); %0.5*(this.modelParamsMax(this.LABELS{in}) + this.modelParamsMin(this.LABELS{in}));
%             end
            this.initial = [1e-6 0.5 1000 0.5 0.01];
%             if (~exist('observed', 'var'))
%                 observed = struct('signal', 1.1*this.norm_signal(0:0.001:0.099), 'times', 0:0.001:0.099);
%             end
            if (exist('observed', 'var'))
                assert(isstruct(observed));
                assert(isfield( observed, 'times'));
                assert(isfield( observed, 'signal'));
                this.observed = observed;
            else
                this.observed = struct('signal', zeros(1,100), 'times', 0:0.001:0.099);
            end
            
            % PRIORS
            this.priors    = cell(1,length(this.LABELS));
            this.priors{1} = @(x) lognormpdf(x, this.modelParams('Delta_chi0'), this.modelParams('Delta_chi0'));
            this.priors{2} = @(x) lognormpdf(x, this.modelParams('Hct'),        this.modelParams('Hct'));
            this.priors{3} = @(x) lognormpdf(x, this.modelParams('R2t'),        this.modelParams('R2t'));
            this.priors{4} = @(x) lognormpdf(x, this.modelParams('Y'),          this.modelParams('Y'));
            this.priors{5} = @(x) lognormpdf(x, this.modelParams('zeta'),       this.modelParams('zeta'));
            
            % MODEL
            this.model      = @(t, paramsHat) this.norm_signal(t, paramsHat('R2t'), paramsHat('zeta'), this.inv_tc( paramsHat));
            this.dmodel     = @(t, pvec)      this.norm_signal(t, pvec(3),          pvec(5),           this.inv_dtc(pvec));
            this.residual   = @(   paramsHat) this.model( this.observed.times, paramsHat) - this.observed.signal;
            this.dresidual  = @(   pvec)      this.dmodel(this.observed.times, pvec)      - this.observed.signal;
             
            % POSTERIOR = likelihood * \Pi_p priors_p
            this.posterior  = @(   paramsHat) sum(-0.5*this.residual(paramsHat).^2) ...
                                        + this.priors{1}(paramsHat(this.LABELS{1})) ...
                                        + this.priors{2}(paramsHat(this.LABELS{2})) ...
                                        + this.priors{3}(paramsHat(this.LABELS{3})) ...
                                        + this.priors{4}(paramsHat(this.LABELS{4})) ...
                                        + this.priors{5}(paramsHat(this.LABELS{5}));  
            this.dposterior = @(   pvec)     sum(-0.5*this.dresidual(pvec).^2) ...
                                        + this.priors{1}(pvec(1)) ...
                                        + this.priors{2}(pvec(2)) ...
                                        + this.priors{3}(pvec(3)) ...
                                        + this.priors{4}(pvec(4)) ...
                                        + this.priors{5}(pvec(5));
                                    
            %rand('state',0); randn('state',0); %#ok<RAND>
 		end % Yablonskiy (ctor)   
        
        function p = get.Hct(this)
            p = this.modelParams('Hct');
        end
        
        function p = get.Delta_chi0(this)
            p = this.modelParams('Delta_chi0');
        end
        
        function p = get.Y(this)
            p = this.modelParams('Y');
        end
        
        function p = get.R2t(this)
            p = this.modelParams('R2t');
        end
        
        function p = get.zeta(this)
            p = this.modelParams('zeta');
        end
        
        function domega = inv_tc(this, params)
            
            %% INV_TC = 1/tc is the characteristic rate, inv_tc*t is the scaling parameter of the perturbation theory.
            %  for t < tc, observations are under motional narrowing.
            %  for t > tc, observed slower processes are in the static dephasing regime.
            %  returns NIfTI if Hct, Delta_chi0 or Y are NIfTI
            if (~exist('params', 'var')); params    = this.modelParams; end
            domega = this.GYROMAG * this.B0 * (4/3) * pi * params('Hct') * params('Delta_chi0') .* (1 - params('Y'));
            assert(domega >= 0);
        end % inv_tc
        
        function domega = inv_dtc(this, pvec)
            
            %% INV_TC = 1/tc is the characteristic rate, inv_tc*t is the scaling parameter of the perturbation theory.
            %  for t < tc, observations are under motional narrowing.
            %  for t > tc, observed slower processes are in the static dephasing regime.
            domega = this.GYROMAG * this.B0 * (4/3) * pi * pvec(2) * pvec(1) .* (1 - pvec(4));
            assert(domega >= 0);
        end % inv_tc
        
        function soft = norm_signal(this, t, R2t, zeta, inv_tc)
            
            %% NORM_SIGNAL
            %  Uasge:  soft = norm_signal(t, R2t, zeta, inv_tc)
            %                             ^ required row-vec of sampling times
            %                                ^ intrinsic R2 of tissue, from measured spin echo?
            %                                     ^ measured DBV
            %                                           ^ defer to this.get.inv_tc
            import mlfsl.* mlentropy.*;
            assert(0 ~= exist('t', 'var'));
            assert(all(t >= 0));
            if (size(t,1) > size(t,2)); t = t'; end
            if (~exist('R2t', 'var'))
                R2t = this.modelParams('R2t');
            end
            if (~exist('zeta', 'var'))
                zeta = this.modelParams('zeta');
            end
            if (~exist('inv_tc', 'var'))
                inv_tc = this.inv_tc(this.modelParams);
            end
            soft = exp(-R2t.*t - zeta.*Yablonskiy.fc(inv_tc.*t));
        end % norm_signal
        
        function tr = trace(this)
            
            %% GET.TRACE uses lazy initialization
            if (~isempty(this.trace_internal))
                tr = this.trace_internal;
            else
                s = RandStream('mt19937ar','Seed', 5489);
                RandStream.setDefaultStream(s);
                if (this.burnInTime > 0 || this.thinningMultiple > 1)
                    tr = slicesample(this.initial,this.nsamples,'logpdf', this.dposterior, ...
                                                                'burnin', this.burnInTime, 'thin', this.thinningMultiple);
                else
                    tr = slicesample(this.initial,this.nsamples,'logpdf', this.dposterior);
                end
                this.trace_internal = tr;
            end
        end % trace
        
        function mPH = modelParamsHat(this) 
            
            tr    = this.trace;
            for p = 1:length(    this.LABELS)
                this.modelParams(this.LABELS{p}) = mean(tr(p));
            end
            mPH = this.modelParams;
            fprintf('paramsHat: \t Hct \t Delta_chi0 \t Y \t R2t \t zeta \n \t\t\t %10.5g \t %10.5g \t %10.5g \t %10.5g \t %10.5g \t %10.5g', ...
                     mPH('Hct'), mPH('Delta_chi0'), mPH('Y'), mPH('R2t'), mPH('zeta'));
        end
                
        function [h,simpost] = plotPosterior(this, idx1, idx2, vantage)
            
            %% PLOTPOSTERIOR
            %  Usage:  [h,simpost] = obj.plotPosterior(lbl1, lbl2 [, vantage])
            assert(isnumeric(idx1));
            assert(isnumeric(idx2));
            range1 = linspace(this.modelParamsMin(this.LABELS{idx1}), this.modelParamsMax(this.LABELS{idx1}));
            range2 = linspace(this.modelParamsMin(this.LABELS{idx2}), this.modelParamsMax(this.LABELS{idx2}));
            if (~exist('vantage','var')); vantage = [-110 30]; end
            h = figure;
            mparams = containers.Map(this.modelParamsHat.keys, this.modelParamsHat.values);
            simpost = zeros(length(range1),length(range2));
            for i = 1:length(range1)
                mparams(this.LABELS{idx1}) = range1(i);
                for j = 1:length(range2)
                    mparams(this.LABELS{idx2}) = range2(j);
                    simpost(i,j) = this.posterior(mparams);
                end
            end
            mesh(range1,range2,simpost)
            xlabel(this.LABELS{idx1})
            ylabel(this.LABELS{idx2})
            zlabel('Posterior density')
            view(vantage(1),vantage(2))
        end % plotPosterior
        
        function h = plotTrace(this, lenMovAvg)
            
            %% PLOTTRACE
            %  Usage:  h = obj.plotTrace(length_moving_average)
            h = figure;
            if (0 ~= exist('lenMovAvg','var'))
                movavg = filter( (1/lenMovAvg)*ones(lenMovAvg,1), 1, this.trace);
                subplot(2,1,1)
                plot(movavg(:,1))
                xlabel('Number of samples')
                ylabel(['Moving ' num2str(lenMovAvg) '-average of intercept']);
                subplot(2,1,2)
                plot(movavg(:,2))
                xlabel('Number of samples')
                ylabel(['Moving ' num2str(lenMovAvg) '-average of the slope']);
            else
                subplot(2,1,1)
                plot(this.trace(:,1))
                ylabel('Intercept');
                subplot(2,1,2)
                plot(this.trace(:,2))
                ylabel('Slope');
            end
        end
        
        function h = plotAutoCorr(this)
            F    =  fft(detrend(this.trace,'constant'));
            F    =  F .* conj(F);
            ACF  =  ifft(F);
            ACF  =  ACF(1:21,:);                          % Retain lags up to 20.
            ACF  =  real([ACF(1:21,1) ./ ACF(1,1) ...
                          ACF(1:21,2) ./ ACF(1,2)]);       % Normalize.
            bounds = sqrt(1/this.nsamples) * [2 ; -2];       % 95% CI for iid normal

            labs = {'Sample ACF for intercept','Sample ACF for slope' };
            h = figure;
            for i = 1:2
                subplot(2,1,i)
                lineHandles  =  stem(0:20, ACF(:,i) , 'filled' , 'r-o');
                set(lineHandles , 'MarkerSize' , 4)
                grid('on')
                xlabel('Lag')
                ylabel(labs{i})
                hold('on')
                plot([0.5 0.5 ; 20 20] , [bounds([1 1]) bounds([2 2])] , '-b');
                plot([0 20] , [0 0] , '-k');
                hold('off')
                a  =  axis;
                axis([a(1:3) 1]);
            end
        end
        
        function h = plotHistTrace(this)
            h = figure;
            subplot(1,1,1)
            hist3(this.trace,[25,25]);
            xlabel('Intercept')
            ylabel('Slope')
            zlabel('Posterior density')
            view(-110,30)
        end
        
        function h = plotMarginals(this)
            h = figure;
            subplot(2,1,1)
            hist(this.trace(:,1))
            xlabel('Intercept');
            subplot(2,1,2)
            ksdensity(this.trace(:,2))
            xlabel('Slope');
        end
        
        function h = plotCumSum(this)
            csum = cumsum(this.trace);
            h = figure;
            subplot(2,1,1)
            plot(csum(:,1)'./(1:this.nsamples))
            xlabel('Number of samples')
            ylabel('Means of the intercept');
            subplot(2,1,2)
            plot(csum(:,2)'./(1:this.nsamples))
            xlabel('Number of samples')
            ylabel('Means of the slope');
        end % plotCumSum
 	end 

	methods (Static)
 		% N.B. (Static, Abstract, Access=', Hidden, Sealed) 
        
        function [q,errbnd] = fc(t_over_tc)
            
            %% FC
            %  Usage:  [q,errbnd] = fc(t_over_tc)
            q       = zeros(size(t_over_tc));
            errbnd  = zeros(size(t_over_tc));
            for t   =   1:length(t_over_tc)
                ttc =            t_over_tc(t);
                [q(t),errbnd(t)] = quadrature(@fc_integrand, 0, 1);
            end
            
            function [q,errbnd] = quadrature(fhandle, lim0, lim)
                try
                    feval      = -1;
                    [q,errbnd] = quadgk(fhandle, lim0, lim);
                    if (errbnd > 0.1)
                        ME = MException('mlentropy:QuadratureFailure', ...
                              sprintf('Yablonskiy.fc.quadrature.quadgk:  errbnd->%g',errbnd));
                        throw(ME);
                    end
                catch ME %#ok<NASGU>
                    try
                        errbnd    = -1;
                        [q,feval] = quadl(fhandle, lim0, lim); %#ok<NASGU>
                        if (feval > 1e5)
                            ME1 = MException('mlentropy:QuadratureFailure', ...
                                  sprintf('Yablonskiy.fc.quadrature.quadl:  feval->%g',feval));
                            throw(ME1);
                        end
                    catch ME1 %#ok<NASGU>
                        try
                            errbnd    = -1;
                            [q,feval] = quad(fhandle, lim0, lim); %#ok<NASGU>
                            if (feval > 1e5)
                                ME2 = MException('mlentropy:QuadratureFailure', ...
                                      sprintf('Yablonskiy.fc.quadrature.quad:  feval->%g',feval));
                                throw(ME2);
                            end
                        catch ME2
                            handerror(ME2, sprintf('Yablonskiy.fc.quadrature failed:  errbnd->%g, feval->%g',errbnd,feval)); %#ok<NODEF>
                        end
                    end
                end
            end
            
            function ig = fc_integrand(u)
                
                %% FC_INTEGRAND
                %  row-vec u is created by quadgk
                [j0,ierr] = besselj(zeros(size(u)), 1.5*ttc*u);
                if (0 ~= ierr)
                    throw(MException('mlentropy:FunctionEvalFailure', ...
                          sprintf('fc_integrand.besselj at u(1)->%g, t/t_c->%g returned ierr->%i',u(1),ttc,ierr)));
                end
                ig = (1/3)*(2 + u).*sqrt(1 - u).*(1 - j0)./(u.*u);
            end 
        end % static fc
            
        function [params,pmin,pmax,pindex] = modelParams0
            
            %% MODELPARAMS0
            %  Usage:   [params,pmin,pmax,pindex] = Yablonskiy.modelParams0
            %  Fix:      synch w/ Yablonskiy.LABELS
            import mlentropy.*;
            labels =              {'Delta_chi0', 'Hct', 'R2t', 'Y', 'zeta'};
            for s = 1:length(labels); assert(strcmp(Yablonskiy.LABELS{s}, labels{s})); end % <-- KLUDGE
            
            params = containers.Map(Yablonskiy.LABELS, ...
                                  { 5e-9,         0.445,   100,  0.33, 0.02});
            pmin   = containers.Map(Yablonskiy.LABELS, ...
                                  { 0,            0.2,   0.1,  0,    0.001});
            pmax   = containers.Map(Yablonskiy.LABELS, ...
                                  { 1e-4,         0.6,   1e6,  1,    1});
            pindex = containers.Map(Yablonskiy.LABELS, ...
                                  { 1,            2,     3,    4,    5});  
        end % static modelParams0
        
        function [simpost,trace,modelParamsHat] = demo
            %% Car Experiment Data
            % In some simple problems such as the previous normal mean inference
            % example, it is easy to figure out the posterior distribution in a closed
            % form. But in general problems that involve non-conjugate priors, the
            % posterior distributions are difficult or impossible to compute
            % analytically. We will consider logistic regression as an example. This
            % example involves an experiment to help model the proportion of cars of
            % various weights that fail a mileage test. The data include observations
            % of weight, number of cars tested, and number failed.  We will work with
            % a transformed version of the weights to reduce the correlation in our
            % estimates of the regression parameters.

            % A set of car weights
            weight = [2100 2300 2500 2700 2900 3100 3300 3500 3700 3900 4100 4300]';
            weight = (weight-2800)/1000;     % recenter and rescale
            % The number of cars tested at each weight
            total = [48 42 31 34 31 21 23 23 21 16 17 21]';
            % The number of cars that have poor mpg performances at each weight
            poor = [1 2 0 3 8 8 14 17 19 15 17 21]';
            
            %% Yablonskiy ctor
            obj = mlentropy.Yablonskiy(weight, poor, total);
            
            %% Yablonskiy Regression Model 
            % Yablonskiy regression, a special case of a generalized linear model, 
            % is appropriate for these data since the response variable is binomial.
            % The logistic regression model can be written as:
            % 
            % $$P(\mathrm{failure}) = \frac{e^{Xb}}{1+e^{Xb}}$$
            % 
            % where X is the design matrix and b is the vector containing the model
            % parameters. In MATLAB(R), we can write this equation as:
            % obj.model = @(b,x) exp(b(1)+b(2).*x)./(1+exp(b(1)+b(2).*x));

            %% Priors
            % If you have some prior knowledge or some non-informative priors are
            % available, you could specify the prior probability distributions for the
            % model parameters. For example, in this demo, we use normal priors for the
            % intercept |b1| and slope |b2|, i.e.
            % obj.priors    = cell(1,2);
            % obj.priors{1} = @(b1) normpdf(b1,0,20);    % prior for intercept 
            % obj.priors{2} = @(b2) normpdf(b2,0,20);    % prior for slope

            %% Posterior probability
            % By Bayes' theorem, the joint posterior distribution of the model parameters
            % is proportional to the product of the likelihood and priors. 
            % obj.posterior = @(b) prod(binopdf(poor,total,obj.model(b,weight))) ...  % likelihood
            %                      * obj.priors{1}(b(1)) * obj.priors{2}(b(2));       % priors

            b1 = linspace(-2.5, -1, 50);
            b2 = linspace(3, 5.5, 50);
            simpost = zeros(50,50);
            for i = 1:length(b1)
                for j = 1:length(b2)
                    simpost(i,j) = obj.posterior([b1(i) b2(j)]);
                end;
            end;
            %mesh(b1,b2,simpost)
            %xlabel('Slope')
            %ylabel('Intercept')
            %zlabel('Posterior density')
            %view(-110,30)

            %% Slice Sampling
            % Monte Carlo methods are often used in Bayesian data analysis to summarize
            % the posterior distribution. The idea is that, even if you cannot compute
            % the posterior distribution analytically, you can generate a random sample
            % from the distribution and use these random values to estimate the posterior
            % distribution or derived statistics such as the posterior mean, median,
            % standard deviation, etc. Slice sampling is an algorithm designed to sample
            % from a distribution with an arbitrary density function, known only up to a
            % constant of proportionality -- exactly what is needed for sampling from a
            % complicated posterior distribution whose normalization constant is unknown.
            % The algorithm does not generate independent samples, but rather a Markovian
            % sequence whose stationary distribution is the target distribution.  Thus,
            % the slice sampler is a Markov Chain Monte Carlo (MCMC) algorithm.  However,
            % it differs from other well-known MCMC algorithms because only the scaled
            % posterior need be specified -- no proposal or marginal distributions are
            % needed.
            %
            % This example shows how to use the slice sampler as part of a Bayesian
            % analysis of the mileage test logistic regression model, including generating
            % a random sample from the posterior distribution for the model parameters,
            % analyzing the output of the sampler, and making inferences about the model
            % parameters.  The first step is to generate a random sample.
            init = [1 1];
            nsamples = 1000; %#ok<*PROP>
            trace = slicesample(init,nsamples,'pdf',obj.posterior,'width',[20 2]);

            %% Analysis of Sampler Convergence/Mixing
            %  After obtaining a random sample from the slice sampler, it is important to
            % investigate issues such as convergence and mixing, to determine whether the
            % sample can reasonably be treated as a set of random realizations from the
            % target posterior distribution. Looking at marginal trace plots is the
            % simplest way to examine the output.
            figure
            subplot(2,1,1)
            plot(trace(:,1))
            ylabel('Intercept');
            subplot(2,1,2)
            plot(trace(:,2))
            ylabel('Slope');
            xlabel('Sample Number');

            %% Moving averages
            % It is apparent from these plots is that the effects of the parameter
            % starting values take a while to disappear (perhaps fifty or so samples)
            % before the process begins to look stationary.
            %
            % It is also be helpful in checking for convergence to use a moving window
            % to compute statistics such as the sample mean, median, or standard
            % deviation for the sample.  This produces a smoother plot than the raw
            % sample traces, and can make it easier to identify and understand any
            % non-stationarity.
%             movavg = filter( (1/50)*ones(50,1), 1, trace);
%             subplot(2,1,1)
%             plot(movavg(:,1))
%             xlabel('Number of samples')
%             ylabel('Means of the intercept');
%             subplot(2,1,2)
%             plot(movavg(:,2))
%             xlabel('Number of samples')
%             ylabel('Means of the slope');

            %% Specifying Burn-in
            % Because these are moving averages over a window of 50 iterations, the first
            % 50 values are not comparable to the rest of the plot.  However, the
            % remainder of each plot seems to confirm that the parameter posterior means
            % have converged to stationarity after 100 or so iterations.  It is also
            % apparent that the two parameters are correlated with each other, in
            % agreement with the earlier plot of the posterior density.
            %
            % Since the settling-in period represents samples that can not reasonably be
            % treated as random realizations from the target distribution, it's probably
            % advisable not to use the first 50 or so values at the beginning of the
            % slice sampler's output.  You could just delete those rows of the output,
            % however, it's also possible to specify a "burn-in" period.  This is
            % convenient when a suitable burn-in length is already known, perhaps from
            % previous runs.
            trace = slicesample(init,nsamples,'pdf',obj.posterior, ...
                                                 'width',[20 2],'burnin',50);
            figure
            subplot(2,1,1)
            plot(trace(:,1))
            ylabel('Intercept');
            subplot(2,1,2)
            plot(trace(:,2))
            ylabel('Slope');

            %% Sample Auto-correlation
            % These trace plots do not seem to show any non-stationarity, indicating that
            % the burn-in period has done its job.
            %
            % However, there is a second aspect of the trace plots that should also be
            % explored.  While the trace for the intercept looks like high frequency
            % noise, the trace for the slope appears to have a lower frequency
            % component, indicating there autocorrelation between values at adjacent
            % iterations.  We could still compute the mean from this autocorrelated
            % sample, but it is often convenient to reduce the storage requirements by
            % removing redundancy in the sample.  If this eliminated the
            % autocorrelation, it would also allow us to treat this as a sample of
            % independent values.  For example, you can thin out the sample by keeping
            % only every 10th value.
            trace = slicesample(init,nsamples,'pdf',obj.posterior,'width',[20 2], ...
                                                            'burnin',50,'thin',10);
                                                    
            %%
            % To check the effect of this thinning, it is useful to estimate the sample
            % autocorrelation functions from the traces and use them to check if the
            % samples mix rapidly.
            F    =  fft(detrend(trace,'constant'));
            F    =  F .* conj(F);
            ACF  =  ifft(F);
            ACF  =  ACF(1:21,:);                          % Retain lags up to 20.
            ACF  =  real([ACF(1:21,1) ./ ACF(1,1) ...
                         ACF(1:21,2) ./ ACF(1,2)]);       % Normalize.
            bounds  =  sqrt(1/nsamples) * [2 ; -2];       % 95% CI for iid normal

            labs = {'Sample ACF for intercept','Sample ACF for slope' };
            figure
            for i = 1:2
                subplot(2,1,i)
                lineHandles  =  stem(0:20, ACF(:,i) , 'filled' , 'r-o');
                set(lineHandles , 'MarkerSize' , 4)
                grid('on')
                xlabel('Lag')
                ylabel(labs{i})
                hold('on')
                plot([0.5 0.5 ; 20 20] , [bounds([1 1]) bounds([2 2])] , '-b');
                plot([0 20] , [0 0] , '-k');
                hold('off')
                a  =  axis;
                axis([a(1:3) 1]);
            end

            %% Thinning to Avoid Redundancy
            % The autocorrelation values at the first lag are significant for the
            % intercept parameter, and even more so for the slope parameter.  We could
            % repeat the sampling using a larger thinning parameter in order to reduce
            % the correlation further.  For the purposes of this demo, however, we'll
            % continue to use the current sample.

            %% Inference for the Model Parameters
            % As expected, a histogram of the sample mimics the plot of the posterior
            % density.
            figure
            subplot(1,1,1)
            hist3(trace,[25,25]);
            xlabel('Intercept')
            ylabel('Slope')
            zlabel('Posterior density')
            view(-110,30)

            %% Marginals of the Posterior
            % You can use a histogram or a kernel smoothing density estimate to summarize
            % the marginal distribution properties of the posterior samples.
            figure
            subplot(2,1,1)
            hist(trace(:,1))
            xlabel('Intercept');
            subplot(2,1,2)
            ksdensity(trace(:,2))
            xlabel('Slope');

            %% Descriptive statistics for the Postereior
            % You could also compute descriptive statistics such as the posterior mean or
            % percentiles from the random samples. To determine if the sample size is
            % large enough to achieve a desired precision, it is helpful to monitor the 
            % desired statistic of the traces as a function of the number of samples.
            csum = cumsum(trace);
            figure
            subplot(2,1,1)
            plot(csum(:,1)'./(1:nsamples))
            xlabel('Number of samples')
            ylabel('Means of the intercept');
            subplot(2,1,2)
            plot(csum(:,2)'./(1:nsamples))
            xlabel('Number of samples')
            ylabel('Means of the slope');
            
            %% Final model coefficients
            % In this case, it appears that the sample size of 1000 is more than
            % sufficient to give good precision for the posterior mean estimate.
            modelParamsHat = mean(trace);
            fprintf(1, '%s:  %s \t %s \n       %12.5g \t %12.5g', 'Bhat', 'Intercept', 'Slope', modelParamsHat(1), modelParamsHat(2));
        end % static demo
        
        function demo2
            
            % A set of car weights
            weight = [2100 2300 2500 2700 2900 3100 3300 3500 3700 3900 4100 4300]';
            weight = (weight-2800)/1000;     % recenter and rescale
            % The number of cars tested at each weight
            total = [48 42 31 34 31 21 23 23 21 16 17 21]';
            % The number of cars that have poor mpg performances at each weight
            poor = [1 2 0 3 8 8 14 17 19 15 17 21]';
            
            obj = mlentropy.Yablonskiy(weight, poor, total);
            
            b1 = linspace(-2.5, -1, 50);
            b2 = linspace(3, 5.5, 50);
            
            obj.plotPosterior(b1,b2);
            obj.burnInTime = 0; obj.thinningMultiple = 1;
            obj.plotTrace;
            obj.burnInTime = 50; obj.thinningMultiple = 10;
            obj.plotTrace;
            obj.plotTrace(50);
            obj.plotAutoCorr;
            obj.plotHistTrace;
            obj.plotMarginals;
            obj.plotCumSum;
            
        end % static demo2
        
        function demo3
            
            import mlentropy.*;
            
            obj = mlentropy.Yablonskiy;

            obj.plotPosterior(1,2);
            obj.burnInTime = 0; obj.thinningMultiple = 1;
            obj.plotTrace;
            obj.burnInTime = 50; obj.thinningMultiple = 10;
            obj.plotTrace;
            obj.plotTrace(50);
            obj.plotAutoCorr;
            obj.plotHistTrace;
            obj.plotMarginals;
            obj.plotCumSum;
            
        end % static demo3
 	end 
	%  Created with Newcl by John J. Lee after newfcn by Frank Gonzalez-Morphy 
 end 
