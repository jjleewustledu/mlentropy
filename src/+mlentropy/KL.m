classdef KL 
	%% KL provides methods for estimating the Kullbach-Leibler divergence, or relative entropy,
	%  the conditional entropies & the mutual information
    %  Uses:  diplib
	 
	%% Version $Revision$ was created $Date$ by $Author$  
 	%% and checked into svn repository $URL$ 
 	%% Developed on Matlab 7.10.0.499 (R2010a) 
 	%% $Id$ 
 	%  N.B. classdef (Sealed, Hidden, InferiorClasses = {?class1,?class2}, ConstructOnLoad) 
    %       methods  (Static, Abstract, Access=', Hidden, Sealed) 
	properties (SetAccess='protected')
        p     = [];
        q     = [];
        mask  = [];
		kldiv = [];
		mi    = [];
        dipmi = [];
		hporq = [];
		hp_giveq = [];
		hq_givep = [];
		hp       = [];
		hq       = [];
        N        = 16384;
 	end 

	methods 

 		function this = KL(p, q, mask) 
            
 			%% KL (ctor) 
 			%  Usage:  obj = KL(p, q)
			%          for distributions or images p,q
            assert(nargin > 0);
            this.p = imcast(p,      'double');
            if (~exist('q','var'))
                 q = rand(size(p)) * dipmax(p); 
            end
            this.q = imcast(q,      'double');
            if (~exist('mask', 'var'))
                 mask = ones(size(p)); 
            end
            this.mask = imcast(mask, 'double');
 
            %assert(all(size(p) == size(q)));
            %assert(all(size(p) == size(mask)));
        end % KL (ctor)        
        function d  = kldivergence(this)
            
            %% KLDIV is relative etnropy between p and q.
			%  ln I = H(p) + D(p||U), uniform distribution U.
			%  It is the expected log-likelihood of rejecting alternative hqpothesis q.
			%  Always positive, not symmetric, not a metric.
			%  Multiply by priors p0/q0 to get Bayes factor.  Noninformative priors ->
			%  q0 = 1 - p0 = 0.5.   Then averaged KL distance D(p||q) + D(q||p) = Bayes factor.
			%  Figure 14.7.1.png
			assert(this.pqReady);
			if (isempty(this.kldiv))
            	%w=warning;
	            %warning('off');

                if (~islogical(this.p))
                    this.p = im2uint8(imcast(this.p, 'double'));
                end
                if (~islogical(this.q))
                    this.q = im2uint8(imcast(this.q, 'double'));
                end
	            in1 = dip_image(this.p);
	            in2 = dip_image(this.q);
	            c1  = diphist(in1,'all',this.N);
	            c1  = c1/sum(c1);
	            c2  = diphist(in2,'all',this.N);
	            c2  = c2/sum(c2);
                
	            tmp = c1./c2; %remove /0 and log 0
	            tmp(c2 ==0)=1;
	            tmp(tmp==0)=1;
	            this.kldiv = sum( c1 .* log2(tmp));

	            %warning(w);
			end
			d = this.kldiv;
        end % kldiv        
        function mi = mutualinfo(this)
            
            %% MUTUALINFO, $I(x,y) = \sum_{i,j} p_{ij} ln( p_{ij}/(p_{i.} p{.j}) )$
			%  Figure 14.7.2.png
            assert(this.pqReady);
            if (isempty(this.mi))
                this.mi = this.H_porq - this.H_pgiveq - this.H_qgivep;
            end
            mi = this.mi;
        end        
        function h  = I(this)
            h = this.mutualinfo;
        end        
        function mi = dipmutualinfo(this)
            
            %% DIPMUTUALINFO, $I(x,y) = \sum_{i,j} p_{ij} ln( p_{ij}/(p_{i.} p{.j}) )$
			%  Figure 14.7.2.png
			assert(this.pqReady);
            if (isempty(this.dipmi))
                w=warning;
                warning('off'); %#ok<WNOFF>

                if (~islogical(this.p))
                    this.p = im2uint8(imcast(this.p, 'double'));
                end
                if (~islogical(this.q))
                    this.q = im2uint8(imcast(this.q, 'double'));
                end
                in1 = dip_image(this.p);
                in2 = dip_image(this.q);
                c1  = diphist(in1,'all',this.N);
                c1  = c1/sum(c1);
                c2  = diphist(in2,'all',this.N);
                c2  = c2/sum(c2);

                c12 = dip_image( double(c1)'*double(c2)); %generate outer product
                h2  = diphist2d(c1,c2,[],[],[],this.N,this.N);
                h2  = h2/sum(h2);

                tmp = h2./c12; %remove /0 and log 0
                tmp(c12==0)=1;
                tmp(tmp==0)=1;

                %% \[ \sum_{i,j} p_{i,j} log_2 \frac{p_{i,j}}{p_{i,\cdot} p_{\cdot,j}} \]
                this.mi = sum( h2 .* log2(tmp));

                warning(w);
            end
            mi = this.mi;
        end % mutualinfo        
        function h  = dipI(this)
            h = this.dipmutualinfo;
        end
        function h  = H_porq(this)
	
			%% HPANDQ, or joint entropy, of p .* q
			%
			assert(this.pqReady);
			if (isempty(this.hporq))
            	this.hporq = entropy(this.p .* this.q);
			end
			h = this.hporq;
        end        
        function h  = H(this)
            h = this.H_porq;
        end        
        function h  = H_pgiveq(this)
	
			%% HP_GIVEQ, or conditional entropy of p
			assert(this.pqReady);
			if (isempty(this.hp_giveq))
            	this.hp_giveq = this.H_porq - this.H_q;
			end
			h = this.hp_giveq;
        end        
        function h  = H_qgivep(this)
	
			%% HQ_GIVEP, or conditional entropy of q
			assert(this.pqReady);
			if (isempty(this.hq_givep))
            	this.hq_givep = this.H_porq - this.H_p;
			end
			h = this.hq_givep;
        end        
        function h  = H_p(this)
	
			%% HP, entropy of p
			%  smallest possible compressed message size of p
			assert(~isempty(this.p));
			if (isempty(this.hp))
             	this.hp = entropy(this.p);
			end
			h = this.hp;
        end        
        function h  = H_q(this)
	
			%% HQ, entorpy of q
			%  smallest possible compressed message size of q
			assert(~isempty(this.q));
			if (isempty(this.hq))
            	this.hq = entropy(this.q);
			end
			h = this.hq;
        end        
        function E  = fslS(this,p) %#ok<*MANU>
            
            %% FSLS is exactly FSL's fslstats -e function
            %  Usage:   E = obj.fslS(p)
            
            p    = imcast(p, 'double');
            pnii = mlfourd.NIfTI(p, ['fslS_' mydatetimestr(now)], 'for KL.fslS');
            pnii.save;
            [s,r] = mlbash(['fslstats ' pnii.fileprefix '.nii.gz -e']);
            if (~s); E = str2num(r); %#ok<ST2NM>
            else     E = nan; end
            delete([pnii.fileprefix mlfourd.NIfTId.FILETYPE_EXT]);
        end        
        function E  = dipentropy(this, p)
            
            %% DIPENTROPY is exactly diplib's entropy function
            %  Usage:   E = obj.dipentropy(p)
            
            if (~islogical(p))
              p = im2uint8(imcast(p, 'double'));
            end
            in = dip_image(p);
            c  = diphist(in,[],this.N);
            c  = c/sum(c);
            c(c==0)=[];
            E  = -sum(c .* log2(c));
        end        
        function E  = dipS(this, p)
            E = this.dipentropy(p);
        end        
        function E  = entropy(this, p)
            E = entropy(imcast(p, 'double'));
        end 
        function E  = S(this, p)
            E = this.entropy(p);
        end        
        function str = struct(this)
            [~,map] = this.summary;
            str     = {};
            kys     = map.keys;
            for k = 1:map.length %#ok<FORFLG,PFUNK>
               str.(kys{k}) = map(kys{k}); %#ok<PFBNS,PFPIE>
            end
        end % struct
        function [msg, map] = summary(this, options)
            
            %% SUMMARY returns string msg & containers.Map map
            %  Usage:   [msg, map] = summary(options)
            %                                ^  available:  'no labels'
            if (~exist('options', 'var')); options = ''; end
            keys  = { 'kldiv'         'I' ...
                      'H'             'Hpq'           'Hqp'        'Hp'         'Hq'   };
                     %'fEp'           'fEq'           'dI' }; 
            lbls  = { 'KL divergence' 'mutual info.' ...  
                      'H(p V q)'      'H(p | q)'      'H(q | p)'   'H(p)'       'H(q)' };
                     %'fslentr(p)'    'fslentr(q)'    'dipmutual info.' };
            vals     = zeros(1, numel(keys));
            vals(1)  = this.kldivergence;
            vals(2)  = this.I;
            vals(3)  = this.H;
            vals(4)  = this.H_pgiveq;
            vals(5)  = this.H_qgivep;
            vals(6)  = this.H_p;
            vals(7)  = this.H_q;
            %vals(8)  = this.fslS(this.p);
            %vals(9)  = this.fslS(this.q);
            %vals(10) = this.dipI;
            map      = containers.Map(keys, vals);
            
            msg   = '';
            for k = 1:numel(keys) %#ok<FORPF>
                switch (options)
                    case 'no labels'
                        msg  = [msg sprintf('%g \n', vals(k))];                         
                    otherwise
                        msg  = [msg sprintf('%i \t%5s \t%15s \t\t%g \n', ...
                                              k,   keys{k},lbls{k},vals(k))]; 
                end
            end
        end % summary
 	end % public methods

    methods (Static)        
        function [msg,map] = summarize(img, img0, special)
            
            %% REPORTKL
            %  Usage:  rep = mlentropy.KL.summarize(img, img0)
            %                                       ^    ^ distributions, images, or fileprefixes fp, fp0
            if (~exist('img0', 'var'))
                img0 = img.zeros;
            end
            kl        = mlentropy.KL(img, img0);
            if (exist('special', 'var'))
                [msg,map] = kl.summary(special);
            else
                [msg,map] = kl.summary;
            end
        end
    end
    
    %% PRIVATE
    
	methods (Access='private')
		function tf = pqReady(this)
			if (isempty(this.p))
				throw(MException('mlentropy:KL:ctorErr', 'KL.kldiv.p is empty')); end
			if (isempty(this.q))
				throw(MException('mlentropy:KL:ctorErr', 'KL.kldiv.q is empty')); end
			tf = true;
        end
        function I  = ParseInputs(this, varargin) %#ok<MANU>

            %% PARSEINPUTS is exactly entropy.ParseInputs from the stats toolbox
            iptchecknargin(1,1,nargin,mfilename);
            iptcheckinput(varargin{1},{'uint8','uint16', 'double', 'logical'},...
                          {'real', 'nonempty', 'nonsparse'},mfilename, 'I',1);
            I = varargin{1};
        end
    end
	%  Created with Newcl by John J. Lee after newfcn by Frank Gonzalez-Morphq 
 end 
