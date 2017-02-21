%% SCRIPT: SAMPLE_KERNEL
function If = nonLocalMeans(I, patchSize, filtSigma, patchSigma)
%
% Sample usage of GPU kernel through MATLAB
%
% DEPENDENCIES
%
%  sampleAddKernel.cu
%  Kernel2.cu
%
  
  %% PARAMETERS
  
    dims = size(I);
    m = dims(1);
    n = dims(2);
    
    if m == 64
        threadsPerBlock = [8 8];
    elseif m == 128
        threadsPerBlock = [8 16];
    elseif m == 256
        threadsPerBlock = [16 16];
    else 
        disp('wrong dim')
    end
    

  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  
  
  % KERNEL
  
  k = parallel.gpu.CUDAKernel( 'neighborCubeKernel.ptx', ...
                               'neighborCubeKernel.cu');
  
  numberOfBlocks  = ceil( [m n] ./ threadsPerBlock );
  
  k.ThreadBlockSize = threadsPerBlock;
  k.GridSize        = numberOfBlocks;
  
  %% DATA
  
  A1 = padarray(I, (patchSize-1)./2, 'symmetric');
  sizeIm = size(I);
  A2 = zeros(prod(sizeIm),prod(patchSize)); % neighbors
  
  H = fspecial('gaussian',patchSize, patchSigma); %filter to apply
  H = H(:) ./ max(H(:));
  H = H';
  
  A = gpuArray(A1);
  B = gpuArray(A2);
  
  B = gather( feval(k, A, B, H, m, n, patchSize(1)) );
  
   %% SECOND KERNEL
  k2 = parallel.gpu.CUDAKernel( 'denoisingKernel.ptx', ...
                               'denoisingKernel.cu');
                             
                           
  k2.ThreadBlockSize = threadsPerBlock;
  k2.GridSize        = numberOfBlocks;

  If = reshape(I, [m*n, 1]);
  
  If = gather( feval(k2, If, B, m, n, patchSize(1), filtSigma));
  If = reshape(If, [m n]);
  

  fprintf('...end %s...\n',mfilename);

  end
%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.1 - December 28, 2016
%
% CHANGELOG
%
%   0.1 (Dec 28, 2016) - Dimitris
%       * initial implementation
%
% ------------------------------------------------------------
