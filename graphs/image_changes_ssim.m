%%==========================================================================
% This script loads SSIM (Structural Similarity Index Measure)
% values computed per object class, either aggregated over all
% images or all bounding boxes, and visualizes the distribution
% via box plots. Separate plots are generated for image-level
% and bounding-box-level SSIM statistics, followed by a combined
% layout. All figures are exported as vector graphics (.svg).
%
% Inputs:    
%   - data/eval/changes/ssim_stats.xlsx
%       (Sheet: 'Summary Statistics', 'Detailed SSIM Scores')
%   - data/eval/changes/ssim_stats_bboxes.xlsx
%       (Sheet: 'Summary Statistics', 'Detailed SSIM Scores')
%
% Outputs:
%   - plots/SSIM_Scores_Img.svg        % SSIM distribution for images
%   - plots/SSIM_Scores_BBoxes.svg     % SSIM distribution for bboxes
%   - plots/SSIM_ScoresSSIM_images_boxes_combined.svg
%                                       % Combined figure layout
%
% Dependencies:
%   - basicResize.m          % Utility for resizing figures
%   - basicExportSVG.m       % Utility for exporting figures to SVG
%
% Notes:
%   - SSIM scores are assumed to be within [-1,1]
%   - Class color coding is grouped for visual consistency
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

%% Configuration: Colors and Plot Settings
% Define a color palette for plotting
newcolors = ["#0d88e6" "#00bfa0" "#ea5545" "#ef9b20" "#f46a9b"];
% Common y-axis limits across plots for visual consistency
my_y_lim = [0.35 1];
% Legend entries for defined groups
legend_text = ["AD NoCG" "AD CG" "AAL NoCG" "AAL CG"];
% Line width for box plots
width = 1.5;

%% GET DATA
% Loads SSIM Data from Excel Files
% Paths to SSIM statistics: whole images and bounding boxes
path_ssim_images = 'data/eval/changes/ssim_stats.xlsx';
path_ssim_bboxes = 'data/eval/changes/ssim_stats_bboxes.xlsx';

% Initialize structured container for data
ssim_data = struct();
ssim_data.image = struct();
ssim_data.image.path = path_ssim_images;
ssim_data.bbox = struct();
ssim_data.bbox.path = path_ssim_bboxes;

% Read summary and per-class SSIM scores
fields = string(fieldnames(ssim_data));
for idx = 1:numel(fields)
    ssim_data.(fields(idx)).summary = readtable(ssim_data.(fields(idx)).path,'Sheet','Summary Statistics');
    ssim_data.(fields(idx)).class_scores = readtable(ssim_data.(fields(idx)).path,'Sheet','Detailed SSIM Scores');
end

% Prepare data for plot: Extract per-category SSIM scores for each class
y_images = [];
y_bboxes = [];

for idx = 1:numel(fields)

    class_names = ssim_data.(fields(idx)).summary.Category;
    y = cell(1,numel(class_names));

    for cidx = 1:numel(class_names)
        name = class_names(cidx);

        isMatch = ismember(ssim_data.(fields(idx)).class_scores.Category(:),name);
        foundData = ssim_data.(fields(idx)).class_scores.SSIMScore(isMatch);

        y(cidx) = {foundData};
    end

    if fields(idx) == "image"
        y_images = y;
    elseif fields(idx) == "bbox"
        y_bboxes = y;
    end
end


%% PLOT IMAGE SSIM
% Plot 1: SSIM for Entire Images
fh1 = figure;
ax = gca;

% Tick labels
names = ssim_data.bbox.summary.Category; % X-axis labels from bbox data for consistency

% Assign consistent color coding by group
groupname={'A1','A1','A1','A2','A2', 'A2', 'B1','B1','B1', 'B2','B2','B2'}';
colors = newcolors;
colors = colors(findgroups(groupname));

% Generate box plots
relevant_plots = [];
for idx = 1:numel(names)
    hold on
    bph = boxchart(y_images{idx}/100,'BoxFaceColor',colors(idx),'BoxFaceAlpha',0.6,'XData',idx*ones(numel(y_images{idx}),1));
    bph.BoxMedianLineColor = newcolors(5);
    bph.WhiskerLineColor = colors(idx);
    bph.MarkerColor = colors(idx);
    bph.MarkerStyle = 'none';
    bph.LineWidth = width;

    % Keep every third plot for legend reference (3 entries per group)
    if mod(idx,3) == 0
        relevant_plots(end+1) = bph;
    end
end

% Plot settings
ax.XAxis.Categories = categorical(1:numel(names));
set(ax,"XTickLabel",names)
ylabel("SSIM")
title("SSIM for whole images")
set(ax,'XGrid','off','YGrid','on')
ylim(my_y_lim)
xline(6.5,'-.','HandleVisibility','off');
legend(relevant_plots,legend_text, 'Orientation','horizontal','Location','southoutside')

%% PLOT BBOX SSIM per CLASS
% Plot 2: SSIM for Bounding Boxes
fh2 = figure;
ax = gca;

% Tick labels
names = ssim_data.bbox.summary.Category;

% Assign consistent color coding by group
groupname={'A1','A1','A1','A2','A2', 'A2', 'B1','B1','B1', 'B2','B2','B2'}';
colors = newcolors;
colors = colors(findgroups(groupname));

% Generate box plots
relevant_plots = [];
for idx = 1:numel(names)
    hold on
    bph = boxchart(y_bboxes{idx}/100,'BoxFaceColor',colors(idx),'BoxFaceAlpha',0.6,'XData',idx*ones(numel(y_bboxes{idx}),1));
    bph.BoxMedianLineColor = newcolors(5);
    bph.WhiskerLineColor = colors(idx);
    bph.MarkerColor = colors(idx);
    bph.MarkerStyle = 'none';
    bph.LineWidth = width;
    
    % Keep every third plot for legend reference (3 entries per group)
    if mod(idx,3) == 0
        relevant_plots(end+1) = bph;
    end
end

% Plot settings
ax.XAxis.Categories = categorical(1:numel(names));
set(ax,"XTickLabel",names)
ylabel("SSIM")
title("SSIM for relevant Bounding Boxes")
set(ax,'XGrid','off','YGrid','on')
ylim(my_y_lim)
xline(6.5,'-.','HandleVisibility','off');
legend(relevant_plots,legend_text, 'Orientation','horizontal','Location','southoutside')

%% size
w = 60;
h = 20;

basicResize(fh1,height=10); % SSIM for whole images
basicResize(fh2,height=10); % SSIM for BBoxes

%% save
base = strcat("plots/", "SSIM_Scores");
file_name1 = strcat(base, "_Img");
file_name2 = strcat(base, "_BBoxes");

basicExportSVG(fh1,file_name1)
basicExportSVG(fh2,file_name2)

%% combined plot

ax1 = get(fh1,'CurrentAxes');
ax2 = get(fh2,'CurrentAxes');
hl1 = findobj(fh1,'type','legend');
hl2 = findobj(fh2,'type','legend');


fh_combined= figure;
tcl=tiledlayout(fh_combined,1,2,"TileSpacing","tight");

ax1.Parent=tcl;
ax1.Layout.Tile=1;
ax2.Parent=tcl;
ax2.Layout.Tile=2;

delete(hl1)
delete(hl2)
leg = legend(relevant_plots,legend_text,'Orientation','horizontal');
leg.Layout.Tile = 'south';

close(fh1,fh2)
basicResize(fh_combined,height=10,width=14.4*2+6);
file_name_combo_2 = strcat(base, "SSIM_images_boxes_combined");
basicExportSVG(fh_combined,file_name_combo_2)