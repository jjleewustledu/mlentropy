classdef Bayesian 
	%% BAYESIAN is the abstract base class for Bayesian inference 
	%  Version $Revision$ was created $Date$ by $Author$  
 	%  and checked into svn repository $URL$ 
 	%  Developed on Matlab 7.10.0.499 (R2010a) 
 	%  $Id$ 
 	%  N.B. classdef (Sealed, Hidden, InferiorClasses = {?class1,?class2}, ConstructOnLoad) 

	properties (Abstract)
 		% N.B. (Abstract, Access='private', GetAccess='protected', SetAccess='protected', ... 
 		%       Constant, Dependent, Hidden, Transient) 
        %model
        %modelName
        %modelParams
 	end 

	methods 

 		function this = Bayesian
 			%% BAYESIAN (ctor) 
 			%  Usage:  obj = Bayesian()
			 
 		end % Bayesian (ctor) 
 	end 

	methods (Abstract)
 		% N.B. (Static, Abstract, Access=', Hidden, Sealed) 
 	end 
	%  Created with Newcl by John J. Lee after newfcn by Frank Gonzalez-Morphy 
 end 
