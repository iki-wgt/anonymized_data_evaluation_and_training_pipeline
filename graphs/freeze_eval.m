%%==========================================================================
% This script evaluates object detection performance of models
% with varying configurations of frozen layers. Models are fine-tuned on
% anonymized data and compared to two reference baselines:
%  (i) the original model trained on full data ("Org on Anon"),
%  (ii) a model trained solely on anonymized data ("Anon on Anon").
%
% Detection performance is quantified using mean Average Precision (mAP)
% at IoU thresholds 0.50:0.95. Data are loaded from CSV evaluation logs,
% and the results are visualized as box plots per model configuration.
%
% Inputs:
%   - CSV files containing per-class AP scores, stored in:
%     • data/eval/freeze/<model_id>_eval/
%     • data/eval/org_on_fb_anonymized/
%     • data/eval/fb_on_fb_anonymized/
%
% Output:
%   - plots/tuned_compare_fb.svg  % Box plot comparing model variants
%
% Dependencies:
%   - basicResize.m          % Utility for resizing figures
%   - basicExportSVG.m       % Utility for exporting figures to SVG
%
% Notes:
%   - AP scores are assumed to be reported in the COCO-style metric
%     AP@[.50:.95], and are converted to percentage scale.
%   - Group colors indicate degree and type of layer freezing.
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

%% GET DATA
% Load CSV-based mAP Data for All Models
% Specify the directory containing the CSV files
main_folder = 'data/eval/freeze/';
models = [["org_on_fb_anonymized","Org on Anon"];...
    ["freeze_back", "Back"];...
    ["freeze_back_no_psa", "Back w/o PSA"];...
    ["freeze_neck", "Neck"];...
    ["freeze_neck_head", "Neck + Head"];...
    ["freeze_neck_head_psa","Neck, Head + PSA"];...
    ["freeze_nothing", "w/o Freezing"];
    ["fb_anonymized_coco","Anon on Anon"]];

% Paths to baseline models
base_model_path = 'data/eval/org_on_fb_anonymized/org_yolov10m_eval';
anonym_model_path = 'data/eval/fb_on_fb_anonymized/fb_yolov10m_eval';

% Initialize container for evaluation data
allModels = struct();

% Loop over all defined model configurations
for idx = 1:size(models, 1)
    eval_for = models(idx, 1);
    model_description = models(idx, 2);

    % Determine folder path depending on baseline status
    if model_description == "Org on Anon"
        csvFolder = base_model_path;
    elseif model_description == "Anon on Anon"
        csvFolder = anonym_model_path;
    else
        csvFolder = fullfile(main_folder, strcat(eval_for, "_eval/"));
    end

    fprintf('Processing Model: %s (%s)\n', eval_for, model_description);

    % Check if the folder exists
    if ~isfolder(csvFolder)
        fprintf('  Folder does not exist: %s\n', csvFolder);
        continue; % Skip to the next iteration if folder is missing
    end

    % Find all CSV files in the folder
    csvFiles = dir(fullfile(csvFolder, '*.csv'));

    % Initialize an empty struct to hold the tables for this model
    allTables = struct();

    % Loop over each CSV file and read it into a table
    for i = 1:numel(csvFiles)
        fileName = csvFiles(i).name;
        filePath = fullfile(csvFolder, fileName);

        try
            % Read the CSV file into a table
            tableData = readtable(filePath);

            % Generate a valid field name by removing the file extension
            [~, fieldName, ~] = fileparts(fileName);
            fieldName = matlab.lang.makeValidName(fieldName); % Ensures the field name is valid

            % Assign the table to the struct with the field name
            allTables.(fieldName) = tableData;

        catch ME
            fprintf('  Failed to load %s: %s\n', fileName, ME.message);
        end
    end

    % Store the model's description and its tables in allModels
    allModels(idx).id = eval_for;
    allModels(idx).name = model_description;
    allModels(idx).data = allTables;
end

% Display loaded model summary
fprintf('\nSummary of Loaded Models:\n');
for idx = 1:length(allModels)
    fprintf('Model %d: %s\n', idx, allModels(idx).name);
    tableNames = fieldnames(allModels(idx).data);
    fprintf('  Tables Loaded (%d): %s\n', length(tableNames), strjoin(tableNames, ', '));
end

% Cleanup intermediate variables
clear allTables; clear csvFiles; clear csvFolder; clear eval_for; clear fieldName;
clear fileName; clear filePath; clear i; clear idx; clear model_description;
clear tableData; clear tableNames;

%% PLOT mAP
fh = figure;
ax = gca;

% Define group colors
newcolors = ["#54504c", "#0d88e6", "#5ad45a", "#ef9b20"];
special_color = "#ea5545";

% Extract AP scores for each model (per class)
class_AP = NaN(size(allModels,2),numel(allModels(1).data.class_AP.class_id));
names = [allModels(:).name];

for idx = 1:size(allModels,2)
    class_AP(idx,:) = allModels(idx).data.class_AP.AP__IoU_0_50_0_95_area_all_maxDets_100_ *100;
end

% Grouping definition for coloring
groupname={'A','B','B','C','C','C','D','A'}';
colors = newcolors(findgroups(groupname));

% Generate box plots
[N,M] = size(class_AP);
for idx = 1:N
    hold on
    bph = boxchart(class_AP(idx,:),'BoxFaceColor',colors(idx),'BoxFaceAlpha',0.6,'XData',idx*ones(M,1));
    bph.BoxMedianLineColor = special_color;
    bph.WhiskerLineColor = colors(idx);
    bph.LineWidth = 1.5;
end

% Axes and figure formatting
ax.XAxis.Categories = categorical(1:N);
set(ax,"XTickLabel",names)
ylabel("AP_{50:95}")
title("Comparision Tuned Models to Base Model")
set(ax,'XGrid','off','YGrid','on')

width = 1.5;
fontsize = 9;
x_pos = .005;
y_pos = .97;

% Highlight baseline split
hxl1 = xline(1.5,'-.','Color',special_color,'LineWidth',width,'LabelOrientation','horizontal','HandleVisibility','off');
text(x_pos,y_pos,'Original','FontSize',fontsize,'Color',special_color,'Units','normalized')
text(x_pos+.15,y_pos,'Anonymized Data','FontSize',fontsize,'Color',special_color,'Units','normalized')

%% size
basicResize(fh,height=10);

%% save
file_name = strcat("plots/", "tuned_compare_fb");
basicExportSVG(fh,file_name)
