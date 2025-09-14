%%==========================================================================
% This script analyzes the influence of ground truth adjustments
% on object detection performance when evaluating models on
% anonymized images. Specifically, it compares the detection 
% performance (AP@[.50:.95]) when:
%     (1) original annotations (from non-anonymized data) are used,
%     (2) relabeled annotations adjusted for anonymized content.
% The evaluation is performed across different model sizes and 
% under two image resolutions: COCO-style and HD (high resolution).
%
% Evaluation Setup:
%   - Models trained on either original (org) or anonymized (fb) data.
%   - Evaluations conducted using two ground truth variants, including
%     combinations with/without relabeling and different resolutions.
%   - Class-wise AP differences (relabeled vs. non-relabeled) are visualized.
%
% Inputs:
%   - CSV files for each YOLOv10 model variant under:
%       data/eval/[training_data]_[eval_condition]/[model_name]_eval/*.csv
%     where:
%       training_data ∈ {"org", "fb"}
%       eval_condition ∈ {"on_own", "on_own_anonymized", etc.}
%
% Outputs:
%   - plots/relabel_changes_own_dataset.svg
%   - plots/relabel_changes_own_dataset_hd.svg
%     → bar plots of class-wise AP changes due to relabeling, for COCO and HD resolution.
%
% Dependencies:
%   - basicExportSVG.m       % For saving plots as vector graphics
%
% Notes:
%   - AP values are assumed to be in COCO-style metric: AP@[.50:.95].
%   - Models are YOLOv10 variants (sizes n, s, m, l, x).
%   - Only selected object classes are visualized.
%
% Author:    Sarah Weiß
%            @ Institute for Artificial Intelligence,
%              Ravensburg-Weingarten University of Applied Sciences
%            @ https://github.com/iki-wgt or https://github.com/Fox93
% Date:      03/15/2025
%==========================================================================

%% Colors
% Define a consistent color palette for per-class bar plots
newcolors = ["#ea5545" "#0d88e6" "#54504c" "#5ad45a" "#ef9b20" "#f46a9b" "#f46a9b" "#f46a9b", "#f46a9b" ,"#f46a9b"];

%% GET OWN DATA
% Load per-model detection results (AP@[.50:.95]) from CSVs across:
% - model sizes ∈ {n, s, m, l, x}
% - training data ∈ {org, fb}
% - evaluation settings: relabeled, non-relabeled, COCO-res, HD-res

% Specify the directory containing the CSV files
main_folder = 'data/eval/';
model_sizes = ["n" "s" "m" "l" "x"]; % -> midx
trained_on = ["org" "fb"]; % -> tidx
eval_on = ["on_own" "on_own_hd" "on_own_anonymized" "on_own_anonymized_hd" "on_own_anonymized_no_relabel" "on_own_anonymized_no_relabel_hd"]; % -> eidx

% Initialize containers for models
org_Models = struct();
fb_Models = struct();

% Load model results into hierarchical structs by training/eval setting
for tidx = 1:numel(trained_on)
    for eidx = 1:numel(eval_on)

        allModels = struct();

        for midx = 1: 1:numel(model_sizes)

            csvFolder = strcat(main_folder,trained_on(tidx),"_",eval_on(eidx),"/",trained_on(tidx),"_yolov10",model_sizes(midx),"_eval");
            if ~isfolder(csvFolder)
                fprintf('  Folder does not exist: %s\n', csvFolder);
                continue; % Skip to the next iteration if folder is missing
            end

            csvFiles = dir(fullfile(csvFolder, '*.csv')); % All CSV files

            % Loop over each CSV file and read it into a table
            allTables = struct();
            for i = 1:numel(csvFiles)
                fileName = csvFiles(i).name;
                filePath = fullfile(csvFolder, fileName);

                try
                    tableData = readtable(filePath);
                    [~, fieldName, ~] = fileparts(fileName);
                    fieldName = matlab.lang.makeValidName(fieldName); % Ensures the field name is valid

                    % Assign the table to the struct with the field name
                    allTables.(fieldName) = tableData;

                catch ME
                    fprintf('  Failed to load %s: %s\n', fileName, ME.message);
                end

            end

            allModels.(model_sizes(midx)) = allTables;
        end

        % Store the model's description and its tables
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

% Display summary of loaded data
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

%% Separate COCO-style and HD resolution evaluations
% Splits data into two structs: one for HD evaluations and one for standard resolution

org_Models_hd = struct();
org_Models_hd.name = "org";
fb_Models_hd = struct();
fb_Models_hd.name = "fb";

for tidx = 1:numel(trained_on)

    use_data = NaN;
    if trained_on(tidx) == "org"
        use_data = org_Models;
    elseif trained_on(tidx) == "fb"
        use_data = fb_Models;
    end

    evals = fieldnames(use_data);
    for eidx = 1:numel(evals(1:end-1))
        if contains(evals(eidx),"hd")
            newFiledString = strrep(evals(eidx), '_hd', '');
            if trained_on(tidx) == "org"
                org_Models_hd.(char(newFiledString)) = use_data.(char(evals(eidx)));
                org_Models = rmfield(org_Models, char(evals(eidx)));
            elseif trained_on(tidx) == "fb"
                fb_Models_hd.(char(newFiledString)) = use_data.(char(evals(eidx)));
                fb_Models = rmfield(fb_Models, char(evals(eidx)));
            end
        end
    end
end

% Clean temporary variables
clear eidx, clear tidx, clear evals, clear use_data

%% PREPARE DATA For PLOT: Changes on classes relabel vs no relabel
% Computes per-class AP differences (relabeled - original GT)
% separately for HD and COCO resolution, for each training origin.

% Evaluation settings: Annoted classes and model size
classes_to_plot = ["person", "backpack", "bottle", "cup", "bowl",  "chair", "bed", "book", "clock", "potted plant"]';
model_size = 'm';
hd_switch = ["hd","not hd"];

% Loop over resolution setting and training source
diffs_hd = NaN(numel(classes_to_plot),numel(trained_on));
diffs = NaN(numel(classes_to_plot),numel(trained_on));
for hdidx = 1:numel(hd_switch)
    for tidx = 1:numel(trained_on)

        use_data = NaN;
        if trained_on(tidx) == "org"
            if hd_switch(hdidx) == "hd"
                use_data = org_Models_hd;
            else
                use_data = org_Models;
            end
        elseif trained_on(tidx) == "fb"
            if hd_switch(hdidx) == "hd"
                use_data = fb_Models_hd;
            else
                use_data = fb_Models;
            end
        end
        
        % Loop over selected object classes
        for cidx = 1:numel(classes_to_plot)

            object = classes_to_plot(cidx);

            isMatch = ismember(use_data.on_own_anonymized.(model_size).class_AP.class_name(:),object);
            isMatch_no_relabel = ismember(use_data.on_own_anonymized_no_relabel.(model_size).class_AP.class_name(:),object);


            if any(isMatch) && any(isMatch_no_relabel)
                AP_object = use_data.on_own_anonymized.(model_size).class_AP.AP__IoU_0_50_0_95_area_all_maxDets_100_(isMatch) *100;
                AP_object_no_relabel = use_data.on_own_anonymized_no_relabel.(model_size).class_AP.AP__IoU_0_50_0_95_area_all_maxDets_100_(isMatch_no_relabel) *100;

                if hd_switch(hdidx) == "hd"
                    diffs_hd(cidx,tidx) = AP_object - AP_object_no_relabel;
                else
                    diffs(cidx,tidx) = AP_object - AP_object_no_relabel;
                end

            else % If class is missing from one version, record as NaN and warn
                disp(strcat('Object ', object, ' not found in one/both of the object lists. Adding a NaN.'));
                if hd_switch(hdidx) == "hd"
                    diffs_hd(cidx,tidx) = Nan;
                else
                    diffs(cidx,tidx) = Nan;
                end
            end
        end
    end
end

% Clean up intermediate variables
clear hdidx, clear tidx, clear cidx, clear usedata, clear object
clear isMatch, clear isMatch_no_relabel, clear AP_object, clear AP_object_no_relabel
clear use_data

%% PLOT: Changes on classes relabel & no relabel

data = {diffs, diffs_hd};
titles = {"Org Model","Anon Model"};

% Plot: Loop over resolution variants
for didx = 1:numel(data)
    use_data = data{didx};

    fh = figure();
    handle_save(didx) = fh;

    th = tiledlayout(fh,1,2);
    th.TileSpacing = 'compact';

    offset = 0.5;
    y_lim = [min(use_data(:))-offset max(use_data(:))+offset];

    for tidx = 1: numel(trained_on)
        nexttile
        
        % Assign color to each bar
        for k = 1:length(use_data(:,tidx))
            bp(k) = bar(classes_to_plot(k),use_data(k,tidx));

            set(bp(k),'facecolor',newcolors(k))
            hold on
        end

        ylim(y_lim)
        title(titles{tidx})
        grid on
    end
    
    % Title of entire figure depending on resolution
    if didx == 1
        title(th,'Changes in AP if relabeled')
    elseif didx == 2
        title(th,'Changes in AP if relabeled HD')
    end
    ylabel(th,'Diffrence of AP_{50:95}')
end


%% Save
base = "plots/";

file_name1 = strcat(base, "relabel_changes_own_dataset");
basicExportSVG(handle_save(1),file_name1)

file_name2 = strcat(base, "relabel_changes_own_dataset_hd");
basicExportSVG(handle_save(2),file_name2)
