%%==========================================================================
% This script analyzes and visualizes detection performance (AP) across
% different object categories stratified by size and frequency in the 
% COCO dataset and its anonymized variant. Models are evaluated across
% combinations of training/evaluation domains, with aggregated and 
% per-class AP metrics reported and visualized via heatmaps.
%
% Main Objectives:
%   - Load per-class AP scores from multiple evaluation runs
%   - Group object classes by [size × frequency] using `MapMap`
%   - Compute average and individual AP metrics (IoU 0.50 and 0.50:0.95)
%   - Compare models trained on original vs. anonymized datasets
%   - Visualize aggregated and differential performance in heatmaps
%   - Highlight SSIM-sensitive classes in tabular summaries
%
% Inputs:
%   - data/eval/<model_id>_eval/*.csv
%       (mAP evaluation results per class per model)
%
% Outputs:
%   - plots/influende_size_frequency_50_90_avg.pdf
%   - plots/influende_size_frequency_50_90_detailed.pdf
%   - plots/influende_size_frequency_50_90_diff_avg.pdf
%   - plots/size_frequency_eval_combined.pdf/svg
%
% Dependencies:
%   - MapMap.m              % Metadata-based class grouping
%   - slanCM.m              % Colormap generator @ https://de.mathworks.com/matlabcentral/fileexchange/120088-200-colormap
%   - heatmap.m             % Customizable Heat Maps @ https://de.mathworks.com/matlabcentral/fileexchange/24253-customizable-heat-maps
%   - basicExportPDF.m      % Export utility (PDF vector format)
%   - basicExportSVG.m      % Export utility (SVG vector format)
%   - basicResize.m         % Resize figures for publication
%
% Notes:
%   - Models: 'org' (trained on original COCO), 'fb' (fine-tuned on anonymized)
%   - Evaluation domains: 'on_org_coco', 'on_fb_anonymized'
%   - Class grouping is fixed: 3×3 grid of [size × frequency]
%   - SSIM-related classes are separately tracked for correlation
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

%% Configuration
% Define model identifiers and human-readable names
models = ["org_on_org_coco" "Org on Org";...
    "org_on_fb_anonymized" "Org on Anon";...
    "fb_on_org_coco" "Anon on Org";...
    "fb_on_fb_anonymized", "Anon on Anon"];

% Set colormap
use_colormap = 'Blues';
new_map = flipud(slanCM(use_colormap));

%% Define object class groups by frequency and size
sizes = ["small","medium","large"];
frequencies = ["low","medium","high"];

% Group objects into 3x3 bins
class_map = MapMap();
class_map.add(["banana", "vase", "bird", "toothbrush"],frequency='low',size='small')
class_map.add(["fire hydrant", "microwave" , "toilet", "keyboard"],frequency='low',size='medium')
class_map.add(["refrigerator", "elephant" , "bed" , "stop sign"],frequency='low',size='large')

class_map.add(["book", "bowl", "baseball glove", "clock"],frequency='medium',size='small')
class_map.add(["dog", "laptop", "baseball bat", "suitcase"],frequency='medium',size='medium')
class_map.add(["couch", "bus", "horse", "motorcycle"],frequency='medium',size='large')

class_map.add(["sports ball", "cup", "cell phone", "bottle"],frequency='high',size='small')
class_map.add(["backpack", "chair", "umbrella", "bench"],frequency='high',size='medium')
class_map.add(["dining table", "car", "truck", "surfboard"],frequency='high',size='large')

class_map.additional = ["baseball glove", "tennis racket"];
%% Load data from CSV
% Load per-class AP results

main_folder = 'data/eval/';
model_sizes = ["m"]; % -> midx
trained_on = ["org"; "fb"]; % -> tidx
eval_on = ["on_org_coco" "on_fb_anonymized"]; % -> eidx

% Loop over training/evaluation domains
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
            for i = 1:numel(csvFiles)
                fileName = csvFiles(i).name;
                filePath = fullfile(csvFolder, fileName);

                try
                    tableData = readtable(filePath);
                    [~, fieldName, ~] = fileparts(fileName);
                    fieldName = matlab.lang.makeValidName(fieldName); % Ensures the field name is valid
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

%% Prepare Data - avg
% Compute aggregated AP metrics (averaged across classes per bin)
common_classes_50_avg = struct();
common_classes_50_95_avg = struct();
for tidx = 1:numel(trained_on)
    for eidx = 1:numel(eval_on)

        model_name = trained_on(tidx) + "_" + eval_on(eidx);
        use_data = NaN;
        if trained_on(tidx) == "org"
            use_data = org_Models;
        elseif trained_on(tidx) == "fb"
            use_data = fb_Models;
        end

        data_50 = NaN(numel(sizes),numel(frequencies));
        data_50_95 = NaN(numel(sizes),numel(frequencies));
        for sidx = 1:numel(sizes)

            col_50 = NaN(numel(frequencies),1);
            col_50_95 = NaN(numel(frequencies),1);

            for fidx = 1:numel(frequencies)

                class_names = class_map.get(size=sizes(sidx),frequency=frequencies(fidx));
                isMatch = ismember(use_data.(eval_on(eidx)).(model_sizes).class_AP.class_name(:),class_names);

                class_mAP_50_95 = mean(use_data.(eval_on(eidx)).(model_sizes).class_AP.AP__IoU_0_50_0_95_area_all_maxDets_100_(isMatch));
                classs_mAP_50 = mean(use_data.(eval_on(eidx)).(model_sizes).class_AP.AP__IoU_0_50_area_all_maxDets_100_(isMatch));

                col_50(fidx) = classs_mAP_50;
                col_50_95(fidx) = class_mAP_50_95;
            end

            data_50(:,sidx) = col_50;
            data_50_95(:,sidx) = col_50_95;

        end

        common_classes_50_avg.(model_name) = data_50*100;
        common_classes_50_95_avg.(model_name) = data_50_95*100;

    end
end

% Clean temporary variables
clear eidx, clear tidx, clear fidx, clear sidx, clear cidx, clear class_names
clear model_name, clear use_data, clear col_50, clear col_50_95, clear data_50, clear data_50_95
clear class_names, clear isMatch, clear class_mAP_50_95, clear classs_mAP_50

%% Prepare Data
% Compute detailed (per-class) AP metrics in each bin
common_classes_50 = struct();
common_classes_50_95 = struct();
for tidx = 1:numel(trained_on)
    for eidx = 1:numel(eval_on)

        model_name = trained_on(tidx) + "_" + eval_on(eidx);
        use_data = NaN;
        if trained_on(tidx) == "org"
            use_data = org_Models;
        elseif trained_on(tidx) == "fb"
            use_data = fb_Models;
        end

        data_50 = NaN(numel(sizes),numel(frequencies)*4);
        data_50_95 = NaN(numel(sizes),numel(frequencies)*4);
        for sidx = 1:numel(sizes)

            col_50 = NaN(numel(frequencies),4);
            col_50_95 = NaN(numel(frequencies),4);

            for fidx = 1:numel(frequencies)

                class_names = class_map.get(size=sizes(sidx),frequency=frequencies(fidx));
                isMatch = ismember(use_data.(eval_on(eidx)).(model_sizes).class_AP.class_name(:),class_names);

                class_mAP_50_95 = use_data.(eval_on(eidx)).(model_sizes).class_AP.AP__IoU_0_50_0_95_area_all_maxDets_100_(isMatch)';
                classs_mAP_50 = use_data.(eval_on(eidx)).(model_sizes).class_AP.AP__IoU_0_50_area_all_maxDets_100_(isMatch)';

                col_50(fidx,:) = classs_mAP_50;
                col_50_95(fidx,:) = class_mAP_50_95;
            end

            data_50(:,(4*(sidx-1)+1:4*(sidx-1)+4)) = col_50;
            data_50_95(:,(4*(sidx-1)+1:4*(sidx-1)+4)) = col_50_95;

        end

        common_classes_50.(model_name) = data_50*100;
        common_classes_50_95.(model_name) = data_50_95*100;

    end
end

% Clean temporary variables
clear eidx, clear tidx, clear fidx, clear sidx, clear cidx, clear class_names
clear model_name, clear use_data, clear col_50, clear col_50_95, clear data_50, clear data_50_95
clear class_names, clear isMatch, clear class_mAP_50_95, clear classs_mAP_50

%% PLOT - avg AP
% Generate AP heatmaps (averaged)
% fh1: 2x2 heatmaps for each model/eval config, showing AP@0.5:0.95
fh1 = figure;
th = tiledlayout(2,2);
th.TileSpacing = 'compact';
for tidx = 1:numel(trained_on)
    for eidx = 1:numel(eval_on)
        nexttile
        model_name = trained_on(tidx) + "_" + eval_on(eidx);
        ismatch = ismember(models(:,1),model_name);
        plot_name = models(ismatch,2);

        heatmap(flip(common_classes_50_95_avg.(model_name)),'','','%0.1f');
        colormap (new_map(60:end,:))

        ax = gca;
        ax.XTick = [1 2 3];
        ax.XTickLabel = ["small","medium","large"];

        ax.YTick = [1 2 3];
        ax.YTickLabel = ["high","med.","low"];

        title(plot_name)

        width = 1.5;
        xlines = [1.5 2.5];
        ylines = [1.5 2.5];
        for idx = 1:numel(xlines)
            xline(xlines(idx),'Color','black','LineWidth',width,'LabelOrientation','horizontal','HandleVisibility','off');
            yline(ylines(idx),'Color','black','LineWidth',width,'LabelOrientation','horizontal','HandleVisibility','off');
        end

    end
end
xlabel(th,"Size")
ylabel(th,"Frequency")
title(th,"AP over Size and Frequency (averaged)")

cbh = colorbar;
cbh.Layout.Tile = 'east';
cbh.FontSize = 10;


%% PLOT AP detailed
% Generate detailed AP heatmaps (per-class)
% fh2: Shows 3x4 layouts of detailed per-class APs
fh2 = figure;
th = tiledlayout(2,2);
th.TileSpacing = 'compact';
for tidx = 1:numel(trained_on)
    for eidx = 1:numel(eval_on)
        nexttile
        model_name = trained_on(tidx) + "_" + eval_on(eidx);
        ismatch = ismember(models(:,1),model_name);
        plot_name = models(ismatch,2);

        heatmap(flip(common_classes_50_95.(model_name)),'','','%0.1f');
        colormap (new_map(60:end,:))

        ax = gca;
        ax.XTick = [2.5 6.5 10.5];
        ax.XTickLabel = ["small","medium","large"];

        ax.YTick = [1 2 3];
        ax.YTickLabel = ["high","med.","low"];

        title(plot_name)

        width = 1.5;
        xlines = [4.5 8.5];
        ylines = [1.5 2.5];
        for idx = 1:numel(xlines)
            xline(xlines(idx),'Color','black','LineWidth',width,'LabelOrientation','horizontal','HandleVisibility','off');
            yline(ylines(idx),'Color','black','LineWidth',width,'LabelOrientation','horizontal','HandleVisibility','off');
        end

    end
end
xlabel(th,"Size")
ylabel(th,"Frequency")
title(th,"AP over Size and Frequency")

cbh = colorbar;
cbh.Layout.Tile = 'east';
cbh.FontSize = 10;



%% PLOT Diffs AVG
% Compute and plot AP differences relative to baseline model
% fh3: Difference w.r.t. org_on_org_coco
fh3 = figure;
th = tiledlayout(2,2);
th.TileSpacing = 'compact';
idx_ax=1;
for tidx = 1:numel(trained_on)
    for eidx = 1:numel(eval_on)
        nexttile
        model_name = trained_on(tidx) + "_" + eval_on(eidx);
        ismatch = ismember(models(:,1),model_name);
        plot_name = models(ismatch,2);

        if (tidx == 1 && eidx == 1)
            heatmap(flip(common_classes_50_95_avg.(model_name)),'','','%0.1f');
        else
            base_model_name = trained_on(1) + "_" + eval_on(1);
            ap_diffs = common_classes_50_95_avg.(model_name) - common_classes_50_95_avg.(base_model_name);
            heatmap(flip(ap_diffs),'','','%0.1f');
        end

        ax(idx_ax) = gca;
        ax(idx_ax).XTick = [1 2 3];
        ax(idx_ax).XTickLabel = ["small","medium","large"];

        ax(idx_ax).YTick = [1 2 3];
        ax(idx_ax).YTickLabel = ["high","med.","low"];

        title(plot_name)

        width = 1.5;
        xlines = [1.5 2.5];
        ylines = [1.5 2.5];
        for idx = 1:numel(xlines)
            xline(xlines(idx),'Color','black','LineWidth',width,'LabelOrientation','horizontal','HandleVisibility','off');
            yline(ylines(idx),'Color','black','LineWidth',width,'LabelOrientation','horizontal','HandleVisibility','off');
        end

        idx_ax = idx_ax +1;
    end
end

for aidx = 1 : numel(ax)
    if aidx ==1
        colormap(ax(aidx),colormap([1 1 1])) %white only
    else
        colormap(ax(aidx),new_map(60:end,:))
    end
end

xlabel(th,"Size")
ylabel(th,"Frequency")
title(th,"AP Differences over Size and Frequency (averaged)")

cbh = colorbar;
cbh.Layout.Tile = 'east';
cbh.FontSize = 10;


%% Size

fh1.Position = [1 1 600 350];
fh2.Position = [1 1 1000 350];
fh3.Position = [1 1 600 350];

%% Save
base = "plots/";
file_name1 = strcat(base, "influende_size_frequency_50_90_avg");
file_name2 = strcat(base, "influende_size_frequency_50_90_detailed");
file_name3 = strcat(base, "influende_size_frequency_50_90_diff_avg");

basicExportPDF(fh1,file_name1)
basicExportPDF(fh2,file_name2)
basicExportPDF(fh3,file_name3)

%% Get AP Change for SSIM Score Related Casses
% Extract class-wise APs for SSIM-relevant categories
ssim_classes = ["car" "stop sign" "traffic light" "knife" "bed" "chair" "cow" "umbrella" "bench" "potted plant" "clock" "tv"];

APs_95_50 = NaN(numel(ssim_classes)+1,numel(models(:,2)));
APs_50 = NaN(numel(ssim_classes)+1,numel(models(:,2)));
modelsize = "m";


for tidx = 1:numel(trained_on)
    for eidx = 1:numel(eval_on)

        model_name = trained_on(tidx) + "_" + eval_on(eidx);
        use_data = NaN;
        if trained_on(tidx) == "org"
            use_data = org_Models;
        elseif trained_on(tidx) == "fb"
            use_data = fb_Models;
        end

        ismatch = ismember(models(:,1),model_name);
        modelidx = find(models(:,1)==model_name,1,"first");
        modeltype = models(ismatch,2);
        APs_95_50(1,modelidx) = modeltype;
        APs_50(1,modelidx) = modeltype;

        for cidx = 1:numel(ssim_classes)

            isMatch = ismember(use_data.(eval_on(eidx)).(modelsize).class_AP.class_name(:),ssim_classes(cidx));

            APs_95_50(cidx+1,modelidx) = use_data.(eval_on(eidx)).(modelsize).class_AP.AP__IoU_0_50_0_95_area_all_maxDets_100_(isMatch)';
            APs_50(cidx+1,modelidx) = use_data.(eval_on(eidx)).(modelsize).class_AP.AP__IoU_0_50_area_all_maxDets_100_(isMatch)';
        end
    end
end

APs_95_50 = APs_95_50(2:end,:) * 100;
APs_50 = APs_50(2:end,:) * 100;

APs_95_50_diffs = [APs_95_50(2:end,1) APs_95_50(2:end,2)-APs_95_50(2:end,1) APs_95_50(2:end,3)-APs_95_50(2:end,1) APs_95_50(2:end,4)-APs_95_50(2:end,1)];
APs_50_diffs = [APs_50(2:end,1) APs_50(2:end,2)-APs_50(2:end,1) APs_50(2:end,3)-APs_50(2:end,1) APs_50(2:end,4)-APs_50(2:end,1)];

APs_95_50_diffs = round(APs_95_50_diffs,1);
APs_50_diffs = round(APs_50_diffs,1);

APs_95_50_diffs(end+1,:) = NaN;
APs_50_diffs(end+1,:) = NaN;
APs_95_50_diffs(2:end,:) = APs_95_50_diffs(1:end-1,:);
APs_50_diffs(2:end,:) = APs_50_diffs(1:end-1,:);

%% check on tie class
tie_idx = 28;
org_tie_AP_50_95 = org_Models.on_org_coco.m.class_AP.AP__IoU_0_50_0_95_area_all_maxDets_100_(28);
fb_tie_AP_50_95 = org_Models.on_fb_anonymized.m.class_AP.AP__IoU_0_50_0_95_area_all_maxDets_100_(28);

diff_tie = fb_tie_AP_50_95 - org_tie_AP_50_95;

%% combined plots
% Generate combined plot layout (aggregated and diff heatmaps)
% Merge tiles from fh1 and fh3 into a new figure
% Access the subplots from the tiled layouts
ax1 = fh1.Children;  % Get axes handles from first tiled figure
ax2 = fh3.Children;  % Get axes handles from second tiled figure
fh_combined = figure;

% Create a tiled layout with 1x2 configuration
tcl = tiledlayout(fh_combined, 1, 2, 'TileSpacing', 'tight');

% Transfer subplots from the first figure
for i = 1:numel(ax1)
    ax1(i).Parent = tcl;
    ax1(i).Layout.Tile = i;
end
% somehow colors where not kept from first figure ...
colormap(fh_combined,new_map(60:end,:))

% Transfer subplots from the second figure
for i = 1:numel(ax2)
    ax2(i).Parent = tcl;
    ax2(i).Layout.Tile = numel(ax1) + i;
end

close(fh1,fh2)
basicResize(fh_combined,height=10,width=14.4*2+6);
file_name_combo_2 = strcat(base, "size_frequency_eval_combined");
basicExportPDF(fh_combined,file_name_combo_2)
basicExportSVG(fh_combined,file_name_combo_2)