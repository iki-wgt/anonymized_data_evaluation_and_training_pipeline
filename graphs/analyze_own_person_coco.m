%%==========================================================================
% This script visualizes the number of images per object class that 
% co-occur with the "person" class in the COCO-derived dataset used 
% for fine-tuning on anonymized data. It compares the original dataset 
% with the anonymized subset depending on the selected mode.
%
% Modes:
%   - 'plain'        % Basic plot with original COCO data
%   - 'borders'      % Visualization of defined frequency borders
%   - 'person_own'   % Overlay: Original vs. Anonymized (fine-tuning) data
%
% Inputs:
%   - data/eval/yolo_10_anonym_eval.xlsx
%       (Sheet: 'Common Objects', Range: A4:G86)
%
% Outputs:
%   - plots/plain_common_classes.svg          % If mode == 'plain'
%   - plots/borders_common_classes.svg        % If mode == 'borders'
%   - plots/person_own_common_classes.svg     % If mode == 'person_own'
%
% Dependencies:
%   - basicResize.m           % Utility for resizing figures
%   - basicExportSVG.m        % Utility for exporting figures to SVG
%
% Notes:
%   - File yolo_10_anonym_eval.xlsx contains list of all classes with
%     image counts, extracted from COCO annotations via COCO API
%   - Vertical grid lines annotate frequency-based groupings: high, mid, low
%   - Minor Y-grid ticks enhance interpretability of value scale
%
% Author:    Sarah Weiß
%            @ Institute for Artificial Intelligence,
%              Ravensburg-Weingarten University of Applied Sciences
%            @ https://github.com/iki-wgt or https://github.com/Fox93
% Date:      03/15/2025
%==========================================================================

clc;
clear;
close all;

%% Configuration: Set visualization mode
% 'plain'      = Original dataset only
% 'borders'    = Show defined frequency borders
% 'person_own' = Overlay: Original vs. anonymized (fine-tune) data
mode = 'person_own'; % 'plain' 'borders' 'person_own'

%% GET DATA
% Read tabular data for class-level image statistics
coco_overview = readtable('data/eval/yolo_10_anonym_eval.xlsx','Sheet','Common Objects','Range','A4:G86');

% Rename columns for semantic clarity
coco_overview.Properties.VariableNames{1} = 'cat';
coco_overview.Properties.VariableNames{2} = 'person_ins';
coco_overview.Properties.VariableNames{3} = 'person_img';
coco_overview.Properties.VariableNames{4} = 'full_ins';
coco_overview.Properties.VariableNames{5} = 'full_img';
coco_overview.Properties.VariableNames{6} = 'own_ins';
coco_overview.Properties.VariableNames{7} = 'own_img';

% Extract relevant rows and class labels
labels = coco_overview.cat(3:end);
idx_labels = 1:numel(labels);
val_person = coco_overview.person_img(3:end);
val_own = coco_overview.own_img(3:end);

% Configure axis ticks
yTicks = sort([1400,3200,9000]);
yMinorTicks = sort(500:500:9000);

%% PLOT: Number of images for classes co-occuring with class "person"
fh = figure;
newcolors = ["#0d88e6","#5ad45a","#ea5545"]; % Color palette
colororder(newcolors);

% values
bar_width = 0.7;
switch mode
    case 'plain' % Original dataset only
        bar(idx_labels,val_person, bar_width)
    case 'borders' % Show defined frequency borders
        bar(idx_labels,val_person, bar_width)
    case 'person_own' % Overlay: Original vs. anonymized (fine-tune) data
        bar(idx_labels,val_person, bar_width)
        hold on
        bar(idx_labels,val_own, bar_width)
        legend('Original COCO Dataset','Anonymized subset for finetuning')
end

% Axis and label configuration
ax = gca;
title('Number of images for classes co-occurring with class ''person''')
set(ax, 'xticklabels',labels, 'XTick', idx_labels, 'TickLength', [0 0], 'FontSize', 20)
switch mode
    case 'plain'
        % use default ticks
    otherwise
        set(ax, 'YTick', yTicks)
        ax.GridLineWidth = 2.5;
end
ax.XGrid='off';
ax.YGrid='on';
ax.MinorGridLineWidthMode ="manual";
ax.YMinorGrid='on';
ax.YAxis.MinorTickValues = yMinorTicks;

% X-axis label rotation for readability
xtickangle(90)
ylabel('Number of Images')

% Frequency category separators (f_high, f_mid, f_low)
color = newcolors(3);
width = 2.5;
fontsize = 20;
switch mode
    case 'plain'
        % skip lines
    otherwise

        hxl1 = xline(0,'-.',{'    f_{high}'},'Color',color,'LineWidth',width,'LabelOrientation','horizontal','HandleVisibility','off');
        hxl2 = xline(16.5,'-.',{'   f_{mid}'},'Color',color,'LineWidth',width,'LabelOrientation','horizontal','HandleVisibility','off');
        hxl3 = xline(45.5,'-.',{'   f_{low}'},'Color',color,'LineWidth',width,'LabelOrientation','horizontal','HandleVisibility','off');
        xline(79.5,'-.','Color',color,'LineWidth',width,'HandleVisibility','off');
        xlim([0,79.5])
        
        hxl1.FontSize = fontsize;
        hxl2.FontSize = fontsize;
        hxl3.FontSize = fontsize;
end

%% size
basicResize(fh,height=20,width=60);

%% save
file_name = strcat("plots/", mode, "_common_classes");
basicExportSVG(fh,file_name)
