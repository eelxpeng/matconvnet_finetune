function imdb = setup_data(dataPath)
% prepare data similar to Caffe under <dataPath> directory
% <dataPath>/train.txt
% <dataPath>/test.txt
% <dataPath>/category.txt
% In train.txt and test.txt, each line is a pair: <imagePath> <label>
% label starts from 1
% e.g.: 
% /home/experiment/data/cat001.jpg 2
% /home/experiment/data/dog011.jpg 1
%
% In category.txt, each line is the label text, and first line correspond to label 1
% second line label 2, and so on. For example, in category.txt:
% dog
% cat
% rice
% then, dog is label 1, cat is label 2, rice is label 3, etc.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written by LI, Xiaopeng @ HKUST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% -------------------------------------------------------------------------
%               Load categories metadata
% -------------------------------------------------------------------------

% cats = {meta.synsets(1:1000).WNID} ;
% descrs = {meta.synsets(1:1000).words} ;
category = {};

fid = fopen(fullfile(dataPath,'category.txt'));
tline = fgetl(fid);
while ischar(tline)
    category{end+1} = tline
    tline = fgetl(fid);
end
fclose(fid)

cats = 1:numel(category);
descrs = category;

imdb.classes.name = cats ;
imdb.classes.description = descrs ;

% -------------------------------------------------------------------------
%               Training images
% -------------------------------------------------------------------------

fprintf('searching training images ...\n') ;
names = {} ;
labels = {} ;
fid = fopen(fullfile(dataPath,'train.txt'));

tline = fgetl(fid);
while ischar(tline)
    C = strsplit(tline);
    names{end+1} = C{1};
    labels{end+1} = str2num(C{2});
    tline = fgetl(fid);
end

labels = horzcat(labels{:}) ;

imdb.images.id = 1:numel(names) ;
imdb.images.name = names ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels ;

% -------------------------------------------------------------------------
%                Validation images
% -------------------------------------------------------------------------
fprintf('searching validation images ...\n') ;
names = {} ;
labels = {} ;
fid = fopen(fullfile(dataPath,'test.txt'));

tline = fgetl(fid);
while ischar(tline)
    C = strsplit(tline);
    names{end+1} = C{1};
    labels{end+1} = str2num(C{2});
    tline = fgetl(fid);
end

labels = horzcat(labels{:}) ;


imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 1e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;

% -------------------------------------------------------------------------
%                 Test images
% -------------------------------------------------------------------------


% -------------------------------------------------------------------------
%                 Postprocessing
% -------------------------------------------------------------------------

% sort categories by WNID (to be compatible with other implementations)
[imdb.classes.name,perm] = sort(imdb.classes.name) ;
imdb.classes.description = imdb.classes.description(perm) ;
relabel(perm) = 1:numel(imdb.classes.name) ;

ok = imdb.images.label >=  0 ;
imdb.images.label(ok) = relabel(imdb.images.label(ok)) ;


