classdef Logistic < mlentropy.Bayesian
	%% LOGISTIC implements Bayesian inference with a logistic model:
    % 
    %  $$f(z) = \frac{e^z}{1 + e^z}, z \equiv B_0 + B_1 x_1 + B_2 x_2 + ...$$
    %  This is a generalized linear model. 
    % 
	%  Version $Revision$ was created $Date$ by $Author$  
 	%  and checked into svn repository $URL$ 
 	%  Developed on Matlab 7.10.0.499 (R2010a) 
 	%  $Id$ 
 	%  N.B. classdef (Sealed, Hidden, InferiorClasses = {?class1,?class2}, ConstructOnLoad) 

	properties 
 		% N.B. (Abstract, Access='private', GetAccess='protected', SetAccess='protected', ... 
 		%       Constant, Dependent, Hidden, Transient) 
        designMat   % n x k   numeric arrays
        outcomes    % n x 1   numeric col. vector
        degeneracy  % for binopdf
        priors      % cell of function handles
        posterior   %         function handle
        model       %         function handle
        modelName = 'logistic'
        modelParams % cell of B0, B1, B2, ...
        burnInTime = 50;
        thinningMultiple = 10;
        traceWidth = [20 2];
        intercept = struct('mu',0,'sigma',20);
        slope = struct('mu',0,'sigma',20);
        trace
    end 

    properties (SetAccess='private')
        designMat_internal
        nsamples_internal = 2000;
        trace_internal
    end
    
    properties (Dependent, SetAccess='private')
        nsamples    % rows of designMat
        npredictors % cols of designMat
        bHat
    end
    
    properties (Access='private')
        initial = [1 1];
    end

	methods 

 		function this = Logistic(designMat, outcome, degen, inter, sl) 
 			%% LOGISTIC (ctor) 
 			%  Usage:  obj = Logistic(designMat, binaryoutcomes, degen, intercept, slope)
			%                         ^          ^ numeric, dip_image, NIfTI, NIfTI filenames or cells of the same
			%                         n x k        n x 1
            %                                                  ^ degeneracy for binopdf
            %                                                         ^          ^ structs with mu, sigma for priors
            this = this@mlentropy.Bayesian;
            import mlfourd.*;
            switch (class(designMat))
                case 'char'
                    pnii = NIfTI(designMat);
                    pdat = pnii.img;
                case NIfTI.NIFTI_SUBCLASS
                    pdat = designMat.img;
                case numeric_types
                    pdat = double(designMat);
                otherwise
                    paramError('mlentropy.Logistic.ctor', 'class(designMat)', class(designMat));
            end
            this.designMat = pdat;
            switch (class(outcome))
                case 'char'
                    onii = NIfTI(outcome);
                    odat = onii.img;
                case NIfTI.NIFTI_SUBCLASS
                    odat = outcome.img;
                case numeric_types
                    odat = double(outcome);
                otherwise
                    paramError('mlentropy.Logistic.ctor', 'class(outcome)', class(outcome));
            end
            this.outcomes  = odat;
            assert(isnumeric(degen));
            this.degeneracy = degen;
            if (exist('inter','var')); this.intercept = inter; end
            if (exist('sl',   'var')); this.slope     = sl; end
            
            this.model     = @(b,x) exp(b(1)+b(2).*x)./(1+exp(b(1)+b(2).*x));
            this.priors    = cell(1,2);
            this.priors{1} = @(b1) normpdf(b1,this.intercept.mu,this.intercept.sigma);  % prior for intercept 
            this.priors{2} = @(b2) normpdf(b2,    this.slope.mu,    this.slope.sigma);  % prior for slope
            this.posterior = @(b)  prod(binopdf(this.outcomes,this.degeneracy,this.model(b,this.designMat))) ...   % likelihood
                                   * this.priors{1}(b(1)) * this.priors{2}(b(2));       % priors
                              
            rand('state',0); randn('state',0); %#ok<RAND>
 		end % Logistic (ctor) 
        
        function tr = get.trace(this)
            
            %% GET.TRACE uses lazy initialization
            %if (~isempty(this.trace_internal))
            %    tr = this.trace_internal;
            %else
                if (this.burnInTime > 0 || this.thinningMultiple > 1)
                    tr = slicesample(this.initial,this.nsamples,'pdf',this.posterior,'width',this.traceWidth, ...
                                                                'burnin',this.burnInTime,'thin',this.thinningMultiple);
                else
                    tr = slicesample(this.initial,this.nsamples,'pdf',this.posterior,'width',this.traceWidth);
                end
            %end
        end % get.trace
        
        function n = get.nsamples(this)
            %if (~isempty(this.nsamples_internal))
            %    n = this.nsamples_internal;
            %else
                n = size(this.designMat,1);
            %end
        end
        
        function n = get.npredictors(this)
            n = size(this.designMat,2);
        end
        
        function bHat = get.bHat(this) 
            bHat = mean(this.trace);
            fprintf(1, '%s: \t %s \t %s \n \t\t\t %10.5g \t %10.5g', 'Bhat', 'Intercept', 'Slope', bHat(1), bHat(2));
        end
        
        
        
        
        
        
        function [h,simpost] = plotPosterior(this, b1, b2, vantage)
            
            %% PLOTPOSTERIOR
            %  Usage:  [h,simpost] = obj.plotPosterior(b1, b2 [, vantage])
            if (~exist('vantage','var')); vantage = [-110 30]; end
            h = figure;
            simpost = zeros(length(b1),length(b2));
            for i = 1:length(b1)
                for j = 1:length(b2)
                    simpost(i,j) = this.posterior([b1(i) b2(j)]);
                end;
            end;
            mesh(b1,b2,simpost)
            xlabel('Slope')
            ylabel('Intercept')
            zlabel('Posterior density')
            view(vantage(1),vantage(2))
        end % plotPosterior
        
        function h = plotTrace(this, lenMovAvg)
            
            %% PLOTTRACE
            %  Usage:  h = obj.plotTrace(length_moving_average)
            h = figure;
            if (exist('lenMovAvg','var'))
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
        function [simpost,trace,bHat] = demo
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
            
            %% Logistic ctor
            obj = mlentropy.Logistic(weight, poor, total);
            
            %% Logistic Regression Model 
            % Logistic regression, a special case of a generalized linear model, 
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
            nsamples = 1000;
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
            bHat = mean(trace);
            fprintf(1, '%s:  %s \t %s \n       %12.5g \t %12.5g', 'Bhat', 'Intercept', 'Slope', bHat(1), bHat(2));
        end % demo
        
        function demo2
            
            % A set of car weights
            weight = [2100 2300 2500 2700 2900 3100 3300 3500 3700 3900 4100 4300]';
            weight = (weight-2800)/1000;     % recenter and rescale
            % The number of cars tested at each weight
            total = [48 42 31 34 31 21 23 23 21 16 17 21]';
            % The number of cars that have poor mpg performances at each weight
            poor = [1 2 0 3 8 8 14 17 19 15 17 21]';
            
            obj = mlentropy.Logistic(weight, poor, total);
            
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
 	end 
	%  Created with Newcl by John J. Lee after newfcn by Frank Gonzalez-Morphy 
 end 
