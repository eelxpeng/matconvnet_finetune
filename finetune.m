function [net, info] = finetune_alexnet(dataPath, varargin)
% Fine tuning using pretrained AlexNet
% Put the scripts at the root folder of <matconvnet>
% see setup_data.m for preparing the data
% put pretrained alexnet: imagenet-matconvnet-alex.mat under the same directory.
%===========================================================
% Written by LI, Xiaopeng @ HKUST
%===========================================================

addpath(genpath('examples'));
run(fullfile(fileparts(mfilename('fullpath')),'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'alexnet' ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
opts.expDir = fullfile(vl_rootnn, 'data', ['finetune' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                       Prepare imdb data
% -------------------------------------------------------------------------
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = setup_data(dataPath) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

num_classes = numel(imdb.classes.name);
% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
net = cnn_imagenet_init('model', opts.modelType, ...
                        'batchNormalization', opts.batchNormalization, ...
                        'weightInitMethod', opts.weightInitMethod, ...
                        'networkType', opts.networkType) ;

% load pre-trained alexnet weights
% if do not want to start with the pretrained weight, just comment this block
preNet = load('./imagenet-matconvnet-alex.mat');
net.layers{1}.weights{1} = preNet.params(1).value ; %conv1f
net.layers{1}.weights{2} = preNet.params(2).value ; %conv1b
net.layers{5}.weights{1} = preNet.params(3).value;  %conv2f
net.layers{5}.weights{2} = preNet.params(4).value;  %conv2b
net.layers{9}.weights{1} = preNet.params(5).value;  %conv3f
net.layers{9}.weights{2} = preNet.params(6).value;  %conv3b
net.layers{12}.weights{1} = preNet.params(7).value; %conv4f
net.layers{12}.weights{2} = preNet.params(8).value; %conv4b
net.layers{15}.weights{1} = preNet.params(9).value; %conv5f
net.layers{15}.weights{2} = preNet.params(10).value;%conv5b
net.layers{19}.weights{1} = preNet.params(11).value;%fc6f
net.layers{19}.weights{2} = preNet.params(12).value;%fc6b
net.layers{22}.weights{1} = preNet.params(13).value;%fc7f
net.layers{22}.weights{2} = preNet.params(14).value;%fc7b
net.layers{25}.weights{1} = preNet.params(15).value;%fc8f
net.layers{25}.weights{2} = preNet.params(16).value;%fc8b

% adapt to customized data by removing last two layers (fc8 and softmaxloss)
% and add customized two layers and initialize fc8 weights
net.layers = net.layers(1:end-2);
net.layers{end+1} = struct('type', 'conv', ...
'weights', {{0.005*randn(1,1,4096,num_classes, 'single'), zeros(1,num_classes,'single')}}, ...
'learningRate', [0.005 0.002], ...
'stride', [1 1], ...
'pad', [0 0 0 0]) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = setup_data(dataPath) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Set the class names in the network
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.description ;

% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, net.meta, imdb) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

% Set the image average (use either an image or a color)
net.meta.normalization.averageImage = rgbMean ;
% S=load('averageImage.mat');
% net.meta.normalization.averageImage = S.averageImage ;

% Set data augmentation statistics
[v,d] = eig(rgbCovariance) ;
net.meta.augmentation.rgbVariance = 0.1*sqrt(d)*v' ;
clear v d ;

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainFn = @cnn_train ;
  case 'dagnn', trainFn = @cnn_train_dag ;
end

[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat')

switch opts.networkType
  case 'simplenn'
    save(modelPath, '-struct', 'net') ;
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
bopts.transformation = meta.augmentation.transformation ;

switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(bopts,x,y) ;
  case 'dagnn'
    fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;
end

% -------------------------------------------------------------------------
function [im,labels] = getSimpleNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
%images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
images = imdb.images.name(batch) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  labels = imdb.images.label(batch) ;
end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  labels = imdb.images.label(batch) ;
  inputs = {'input', im, 'label', labels} ;
end

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 101: end);
bs = 256 ;
opts.networkType = 'simplenn' ;
fn = getBatchFn(opts, meta) ;
avg = {}; rgbm1 = {}; rgbm2 = {};

for t=1:bs:numel(train)
  batch_time = tic ;
  batch = train(t:min(t+bs-1, numel(train))) ;
  fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
  temp = fn(imdb, batch) ;
  z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  avg{end+1} = mean(temp, 4) ;
  rgbm1{end+1} = sum(z,2)/n ;
  rgbm2{end+1} = z*z'/n ;
  batch_time = toc(batch_time) ;
  fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
