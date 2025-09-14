%%==========================================================================
% This script benchmarks object detection performance (AP/mAP) across
% multiple YOLOv10 model variants trained on original (COCO) vs.
% anonymized datasets. Performance is evaluated per model size (n–x),
% for all classes and the 'person' class, and compared to published
% baselines and own training configurations.
%
% Main Objectives:
%   - Load and structure AP/mAP evaluation results from CSV exports
%   - Compare performance across training (org/fb) and evaluation domains
%   - Extract and visualize AP values for the 'person' class
%   - Benchmark all variants against published results ([10], [18])
%   - Visualize results as bar charts and box plots
%   - Export individual and combined figures for reporting
%
% Inputs:
%   - data/eval/yolo_10_anonym_eval.xlsx
%       (Paper reference data DOI: 10.1109/ACCESS.2024.3352146)
%   - data/eval/<model_id>_eval/*.csv
%       (Evaluation output from YOLOv10 runs)
%
% Outputs:
%   - plots/mAP_all_Sizes_all_Classes.svg
%   - plots/mAP_all_Sizes_Person.svg
%   - plots/Comp_Source_to_Own_m_Models_50_95_v2.svg
%   - plots/Comp_Source_to_Own_m_Models_50_v2.svg
%   - plots/compare_to_source_both_iou_ranges.svg
%   - plots/combined_mAP_all_Sizes_AP_person.svg
%
% Dependencies:
%   - basicResize.m        % Resizes figure dimensions
%   - basicExportSVG.m     % Exports vector-based SVG figures
%
% Notes:
%   - Model sizes evaluated: n, s, m, l, x
%   - Evaluation metrics: AP@50:95 and AP@50 (COCO style)
%   - Baseline values from DOI: 10.1109/ACCESS.2024.3352146, further called
%   source [10]
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

%% Configuration: Colors and Labels
% Define color palette, legend mappings, and layout settings for plots
newcolors = ["#54504c" "#ea5545" "#0d88e6" "#5ad45a" "#ef9b20" "#f46a9b"];

labels = struct();
labels.base = 'YOLOv10';
labels.org_on_org_coco = 'Org on Org';
labels.org_on_fb_anonymized = 'Org on Anon';
labels.fb_on_org_coco = 'Anon on Org';
labels.fb_on_fb_anonymized = 'Anon on Anon';

% Styling parameters
width = 1.5;
fontsize = 10;
special_color = "#ea5545";
diff_line_pos = 5.5;
x_pos = .01;
y_pos = .97;

%% GET OWN DATA
% Read AP/mAP results from local CSV files for all YOLOv10 model sizes
% and both training/evaluation domain combinations (original vs. anonymized)

% Default original YOLOv10 (Source https://docs.ultralytics.com/models/yolov10/#performance)
YOLOv10_mAP = [39.5 46.8 51.3 53.4 54.4];

% Specify the directory containing the CSV files
main_folder = ['data/eval/'];

model_sizes = ["n" "s" "m" "l" "x"]; % -> midx
trained_on = ["org" "fb"]; % -> tidx
eval_on = ["on_org_coco" "on_fb_anonymized"]; % -> eidx

% Loop over trainging and evaluation configurations
org_Models = struct();
fb_Models = struct();
for tidx = 1:numel(trained_on)
    for eidx = 1:numel(eval_on)

        allModels = struct();

        for midx = 1: 1:numel(model_sizes)

            csvFolder = strcat(main_folder,trained_on(tidx),"_",eval_on(eidx),"/",trained_on(tidx),"_yolov10",model_sizes(midx),"_eval");

            if ~isfolder(csvFolder)
                fprintf('  Folder does not exist: %s\n', csvFolder);
                continue; % Skip to the next iteration if folder is missing
            end

            csvFiles = dir(fullfile(csvFolder, '*.csv'));
            allTables = struct();

            % Loop over each CSV file and read it into a table
            for i = 1:numel(csvFiles)
                fileName = csvFiles(i).name;
                filePath = fullfile(csvFolder, fileName);

                try
                    tableData = readtable(filePath);

                    % Generate a valid field name by removing the file extension
                    [~, fieldName, ~] = fileparts(fileName);
                    fieldName = matlab.lang.makeValidName(fieldName);
                    allTables.(fieldName) = tableData;

                catch ME
                    fprintf('  Failed to load %s: %s\n', fileName, ME.message);
                end

            end

            allModels.(model_sizes(midx)) = allTables;
        end

        if trained_on(tidx) == "org"
            org_Models(tidx).(eval_on(eidx)) = allModels;
        elseif trained_on(tidx) == "fb"
            fb_Models(tidx-1).(eval_on(eidx)) = allModels;
        end
    end

    % Store the model's description and its tables
    if trained_on(tidx) == "org"
        org_Models(tidx).name = trained_on(tidx);
    elseif trained_on(tidx) == "fb"
        fb_Models(tidx-1).name = trained_on(tidx);
    end
end

% Summarize loaded models
fprintf('\n--Summary of Loaded Models--\n');
for tidx = 1:numel(trained_on)
    use_data = NaN;
    if trained_on(tidx) == "org"
        use_data = org_Models;
    elseif trained_on(tidx) == "fb"
        use_data = fb_Models;
    end

    fprintf('Model: %s\n', use_data.name);
    for eidx = 1:numel(eval_on)
        fprintf('\n  Evaluated %s\n', eval_on(eidx));
        tableNames = fieldnames(use_data.(eval_on(eidx)));
        fprintf('  Models loaded (%d): %s\n', length(tableNames), strjoin(tableNames, ', '));
        for idx = 1:numel(tableNames)
            field = tableNames{idx};
            csvNames = fieldnames(use_data.(eval_on(eidx)).(field));
            fprintf('    %s - Tables loaded (%d): %s\n', string(tableNames(idx)), length(csvNames), strjoin(csvNames, ', '));
        end

    end
end

% Clean temporary variables
clear allModels; clear allTables; clear csvNames; clear csvFolder, clear csvFiles;
clear idx; clear midx; clear eidx; clear midx; clear tidx; clear i;
clear field; clear fieldName; clear fileName; clear filePath;
clear tableData; clear tableNames; clear use_data;

%% Load Reference Paper Data
% Load published AP values from [10] to compare with own models
% (sheets: YOLO Size m Classwise, AP@0.50:0.95 and AP@0.50)
paper = struct();
paper.mAP_50_95 = readtable('data/eval/yolo_10_anonym_eval.xlsx','Sheet','YOLO Size m Classwise','Range','B5:G19');
paper.mAP_50 = readtable('data/eval/yolo_10_anonym_eval.xlsx','Sheet','YOLO Size m Classwise','Range','B25:G39');
paper.size = "m";

var_names = string(paper.mAP_50.Properties.VariableNames);
for vidx = 1:numel(var_names)
    if vidx < 3
        continue
    end
    paper.mAP_50_95.(var_names(vidx)) = paper.mAP_50_95.Baseline + paper.mAP_50_95.(var_names(vidx));
    paper.mAP_50.(var_names(vidx)) = paper.mAP_50.Baseline + paper.mAP_50.(var_names(vidx));
end

% Get OWN Data relevant to paper data
class_names = paper.mAP_50_95.Class; % relevant classes

own = struct();
own.mAP_50_95 = table();
own.mAP_50 = table();

for tidx = 1:numel(trained_on)

    use_data = NaN;
    if trained_on(tidx) == "org"
        use_data = org_Models;
    elseif trained_on(tidx) == "fb"
        use_data = fb_Models;
    end

    for eidx = 1:numel(eval_on)
        %get relevant class data
        isMatch = ismember(use_data.(eval_on(eidx)).(paper.size).class_AP.class_name(:),class_names);

        foundClasses = use_data.(eval_on(eidx)).(paper.size).class_AP.class_name(isMatch);
        mAP_50_95 = use_data.(eval_on(eidx)).(paper.size).class_AP.AP__IoU_0_50_0_95_area_all_maxDets_100_(isMatch);
        mAP_50 = use_data.(eval_on(eidx)).(paper.size).class_AP.AP__IoU_0_50_area_all_maxDets_100_(isMatch);
        
        % add classes only once
        if isempty(own.mAP_50_95) && isempty(own.mAP_50)
            own.mAP_50_95.('Class') = foundClasses;
            own.mAP_50.('Class') = foundClasses;
        end

        % add relevant data
        own.mAP_50_95.(strcat(trained_on(tidx),'_',eval_on(eidx))) = mAP_50_95 * 100;
        own.mAP_50.(strcat(trained_on(tidx),'_',eval_on(eidx))) = mAP_50 * 100;
    end
end

% Sort
paper.mAP_50_95 = sortrows(paper.mAP_50_95,'Class');
paper.mAP_50 = sortrows(paper.mAP_50,'Class');
own.mAP_50_95 = sortrows(own.mAP_50_95,'Class');
own.mAP_50 = sortrows(own.mAP_50,'Class');

% Clean temporary variables
clear var_names; clear vidx;
clear foundClasses; clear mAP_50_95, clear mAP_50; clear class_names;
clear use_data; clear eidx; clear tidx; clear isMatch;


%% GET mAP/AP DATA all model sizes & person class
% Aggregate results across model sizes for both mAP and class-specific
% AP (for the 'person' class). Format results for bar chart visualization

% Base Model
x_labels_ids = {'base'};
all_y = YOLOv10_mAP';
all_person_y = NaN;

% Own Models
for tidx = 1:numel(trained_on)

    use_data = NaN;
    if trained_on(tidx) == "org"
        use_data = org_Models;
    elseif trained_on(tidx) == "fb"
        use_data = fb_Models;
    end

    % Gather data based on eval images
    for eidx = 1:numel(eval_on)

        y = NaN(numel(model_sizes),1);
        person_y = NaN(numel(model_sizes),1);

        tableNames = fieldnames(use_data.(eval_on(eidx)));
        for idx = 1:numel(tableNames)
            % Add own data
            field = tableNames{idx};
            m_AP = use_data.(eval_on(eidx)).(field).mAP.AP(1);
            y(idx)=m_AP*100;

            isMatch = ismember(use_data.(eval_on(eidx)).(field).class_AP.class_name(:),'person');
            person_y(idx) = use_data.(eval_on(eidx)).(field).class_AP.AP__IoU_0_50_0_95_area_all_maxDets_100_(isMatch) *100;
        end

        % Data
        all_y(:,end+1) = y;
        if isnan(all_person_y)
            all_person_y = person_y;
        else
            all_person_y(:,end+1) = person_y;
        end

        % ID for Labels
        label_id = strcat(trained_on(tidx),"_",eval_on(eidx));
        x_labels_ids{end+1} = label_id;
    end
end

% Clean temporary variables
clear idx; clear midx; clear eidx; clear midx; clear tidx; clear i;
clear field; clear isMatch; clear use_data;
clear person_y; clear y; clear tableNames;

%% PLOT mAP all Classes all model sizes
% Bar plot showing mAP across model sizes and evaluation configs.
% Includes reference source models for comparison

fh1 = figure;
ax = gca;

all_y(:,3:6) = all_y(:,2:5);
source_y = [33.1 41.4 46.4 48.3 48.4]';
all_y(:,2) = source_y;

bp = bar(all_y(:,3:end)');

% Colors
colororder(newcolors(2:end));

% Labels & Titel
title("Trained Models: mAP for all Classes")
x_labels_ids(3:6) = x_labels_ids(2:end);
x_labels_ids(2) = {"source"};

for idx = 1:numel(x_labels_ids)
    if idx == 1 || idx  == 2
        continue
    end
    tick_labels{idx-2} = labels.(x_labels_ids{idx});
end

xticks(1:numel(tick_labels))
xticklabels(tick_labels)
set(ax,'fontsize',10)
ylabel("mAP_{50:95}")
ylim([0 60])

% Legend
legend(model_sizes,'Location','southoutside','Orientation','horizontal')

% other
set(ax,'XGrid','off','YGrid','on')

val = cat(2,bp.YData);
for idx = 1:numel(val)
    val_str(idx) = string(sprintf('%.1f',val(idx)));
end

text(cat(2,bp.XEndPoints),cat(2,bp.YEndPoints),val_str, 'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize',7)
box off

%% PLOT 'person' AP all model sizes
% Bar plot showing AP@50:95 specifically for the 'person' class
% across all model sizes and training/eval combinations
fh2 = figure;
ax = gca;

bp = bar(all_person_y');

% Colors
colororder(newcolors(2:end));

% Labels & Titel
title("Trained Models: AP for 'person'")

xticks(1:4)
xticklabels(tick_labels)
set(ax,'fontsize',10)
ylabel("AP_{50:95}")
ylim([0 60])

% Legend
legend(model_sizes,'Location','southoutside','Orientation','horizontal')

% other
set(ax,'XGrid','off','YGrid','on')

val = cat(2,bp.YData);
for idx = 1:numel(val)
    val_str(idx) = string(sprintf('%.1f',val(idx)));
end

text(cat(2,bp.XEndPoints),cat(2,bp.YEndPoints),val_str, 'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize',7)
box off

%% PLOT mAP 50..95 PAPER vs OWN V2
% Same as above but excluding the baseline for simplified comparison

fh3_2 = figure;
ax = gca;

newcolors = ["#0d88e6", "#5ad45a", "#5ad45a", "#ea5545"];

names = [paper.mAP_50_95.Properties.VariableNames(2:end) own.mAP_50_95.Properties.VariableNames(2:end)];
names(1) = {'Baseline'};
for lidx = 6:numel(names)
    names(lidx) = {labels.(string(names(lidx)))};
end

groupname={'A2','A2','A2','A2','B2','B2','B2','B2'}';
colors = newcolors;
colors = colors(findgroups(groupname));

data = table2array([ paper.mAP_50_95(:,2:end) own.mAP_50_95(:,2:end)]);
data = data(:,2:end);

[N,M] = size(data');

for idx = 1:N
    hold on
    bph = boxchart(data(:,idx),'BoxFaceColor',colors(idx),'BoxFaceAlpha',0.6,'XData',idx*ones(M,1));
    bph.BoxMedianLineColor = newcolors(end);
    bph.WhiskerLineColor = colors(idx);
    bph.MarkerColor = colors(idx);
    bph.MarkerStyle = 'none';
    bph.LineWidth = width;
end

ylim([0 80])
ax.XAxis.Categories = categorical(1:N);
set(ax,"XTickLabel",names(2:end))
ylabel("AP_{50:95}")
title("Comparison Anonymization Methods")
set(ax,'XGrid','off','YGrid','on')

hxl1 = xline(diff_line_pos-1,'-.','Color',special_color,'LineWidth',width,'LabelOrientation','horizontal','HandleVisibility','off');
text(x_pos,y_pos,'Models of [10]','FontSize',fontsize,'Color',special_color,'Units','normalized')
text(x_pos+.51,y_pos,'Our Models','FontSize',fontsize,'Color',special_color,'Units','normalized')

%% PLOT mAP 50 PAPER vs OWN V2
% Simplified AP@50 comparison excluding baseline column

fh4_2 = figure;
ax = gca;

newcolors = ["#0d88e6", "#5ad45a", "#5ad45a", "#ea5545"];

%class_AP = NaN(size(allModels,2),numel(allModels(1).data.class_AP.class_id));
names = [paper.mAP_50.Properties.VariableNames(2:end) own.mAP_50.Properties.VariableNames(2:end)];

for lidx = 6:numel(names)
    names(lidx) = {labels.(string(names(lidx)))};
end
groupname={'A2','A2','A2','A2','B2','B2','B2','B2'}';
colors = newcolors;
colors = colors(findgroups(groupname));

data = table2array([ paper.mAP_50(:,2:end) own.mAP_50(:,2:end)]);
data = data(:,2:end);

[N,M] = size(data');

for idx = 1:N
    hold on
    bph = boxchart(data(:,idx),'BoxFaceColor',colors(idx),'BoxFaceAlpha',0.6,'XData',idx*ones(M,1));
    bph.BoxMedianLineColor = newcolors(end);
    bph.WhiskerLineColor = colors(idx);
    bph.MarkerColor = colors(idx);
    bph.MarkerStyle = 'none';
    bph.LineWidth = width;
end

ylim([0 80])
ax.XAxis.Categories = categorical(1:N);
set(ax,"XTickLabel",names(2:end))
ylabel("AP_{50}")
title("Comparision Anonymization Methods")
set(ax,'XGrid','off','YGrid','on')

hxl2 = xline(diff_line_pos-1,'-.','Color',special_color,'LineWidth',width,'LabelOrientation','horizontal','HandleVisibility','off');
text(x_pos,y_pos,'Models of [10]','FontSize',fontsize,'Color',special_color,'Units','normalized')
text(x_pos+.51,y_pos,'Our Models','FontSize',fontsize,'Color',special_color,'Units','normalized')

%% size
w = 60;
h = 20;

basicResize(fh1,height=8);
basicResize(fh2,height=8);
basicResize(fh3_2,height=10);
basicResize(fh4_2,height=10);

%% save
base = "plots/";
file_name1 = strcat(base, "mAP_all_Sizes_all_Classes");
file_name2 = strcat(base, "mAP_all_Sizes_Person");
file_name3_2 = strcat(base, "Comp_Source_to_Own_m_Models_50_95_v2");
file_name4_2 = strcat(base, "Comp_Source_to_Own_m_Models_50_v2");

basicExportSVG(fh1,file_name1)
basicExportSVG(fh2,file_name2)
basicExportSVG(fh3_2,file_name3_2)
basicExportSVG(fh4_2,file_name4_2)

%% Cobined plots
% Combine two bar plots into one layout to show overall vs. person-specific AP

ax1 = get(fh3_2,'CurrentAxes');
ax2 = get(fh4_2,'CurrentAxes');

fh_combined_methods= figure;
tcl=tiledlayout(fh_combined_methods,1,2,"TileSpacing","tight");
title(tcl,"Comparision Anonymization Methods")
title(ax1, "IOU of 50 to 95")
title(ax2, "IOU of 50")
ylabel(ax1,"AP_{50:95}")
ylabel(ax2, "AP_{50}")
yticks(ax1,0:10:80)
yticks(ax2,0:10:80)

ax1.Parent=tcl;
ax1.Layout.Tile=1;
ax2.Parent=tcl;
ax2.Layout.Tile=2;

close(fh3_2,fh4_2)
basicResize(fh_combined_methods,height=10,width=14.4*2);
file_name_combo_1 = strcat(base, "compare_to_source_both_iou_ranges");
basicExportSVG(fh_combined_methods,file_name_combo_1)


ax1 = get(fh1,'CurrentAxes');
ax2 = get(fh2,'CurrentAxes');
hl1 = findobj(fh1,'type','legend');
hl2 = findobj(fh2,'type','legend');


fh_combined_sizes= figure;
tcl=tiledlayout(fh_combined_sizes,1,2,"TileSpacing","tight");

title(ax1, "Trained Models: mAP")
title(ax2, "Trained Models: AP for class 'person'")
ylabel(ax1,"mAP_{50:95}")
ylabel(ax2, "AP_{50:95}")
ylim(ax1,[0,60])
ylim(ax2,[0,60])
yticks(ax1,[0:10:60])
yticks(ax2,[0:10:60])

ax1.Parent=tcl;
ax1.Layout.Tile=1;
ax2.Parent=tcl;
ax2.Layout.Tile=2;

delete(hl1)
leg = legend(hl2.String);
leg.Layout.Tile = 'south';

close(fh1,fh2)
basicResize(fh_combined_sizes,height=10,width=14.4*2+6);
file_name_combo_2 = strcat(base, "combined_mAP_all_Sizes_AP_person");
basicExportSVG(fh_combined_sizes,file_name_combo_2)