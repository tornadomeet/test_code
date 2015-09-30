%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% these script is used to convet png with colormap to png without colormap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set you own dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_set = 'H:\\data\\image\\PASCAL_VOC\\VOCdevkit\\VOC2012'
orig_folder = fullfile(data_set, 'SegmentationClassAug_Visualization')
save_folder = fullfile(data_set, 'SegmentationClassAug')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to change values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imgs_dir = dir(fullfile(orig_folder, '*.png'));

if ~exist(save_folder, 'dir')
    mkdir(save_folder)
end

for i = 1 : numel(imgs_dir)
    fprintf(1, 'processing %d (%d) ...\n', i, numel(imgs_dir));
    
    img = imread(fullfile(orig_folder, imgs_dir(i).name));
    
    imwrite(img, fullfile(save_folder, imgs_dir(i).name));  % why no colormap?
end