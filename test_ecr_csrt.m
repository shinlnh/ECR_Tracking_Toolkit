% Test ECR_CSRT tracker on OTB sequences
% Simple test script without full toolkit

clear; clc; close all;

% Add tracker path
addpath('toolkit/trackers/ECR_CSRT');

% OTB dataset path
dataset_root = 'e:\SourceCode\C2P\Project\ECR_Tracking\otb100\OTB-dataset\OTB100';

% Test sequences
test_seqs = {'Basketball', 'Gym'};

for i = 1:length(test_seqs)
    seq_name = test_seqs{i};
    fprintf('\n========================================\n');
    fprintf('Testing sequence: %s\n', seq_name);
    fprintf('========================================\n');
    
    % Load sequence
    seq_path = fullfile(dataset_root, seq_name);
    gt_file = fullfile(seq_path, 'groundtruth_rect.txt');
    img_dir = fullfile(seq_path, 'img');
    
    if ~exist(gt_file, 'file')
        fprintf('Ground truth not found: %s\n', gt_file);
        continue;
    end
    
    % Read ground truth
    gt = dlmread(gt_file);
    seq.name = seq_name;
    seq.init_rect = gt(1, :);
    
    % Get image list
    img_files = dir(fullfile(img_dir, '*.jpg'));
    if isempty(img_files)
        img_files = dir(fullfile(img_dir, '*.png'));
    end
    
    seq.len = length(img_files);
    seq.s_frames = cell(seq.len, 1);
    for j = 1:seq.len
        seq.s_frames{j} = fullfile(img_dir, img_files(j).name);
    end
    
    fprintf('  Frames: %d\n', seq.len);
    fprintf('  Init box: [%.1f, %.1f, %.1f, %.1f]\n', seq.init_rect);
    
    % Run tracker
    try
        result = run_ECR_CSRT(seq, [], false);
        
        fprintf('\n  Results:\n');
        fprintf('    FPS: %.2f\n', result.fps);
        fprintf('    Boxes returned: %d\n', size(result.res, 1));
        
        % Calculate IoU
        if size(result.res, 1) == size(gt, 1)
            ious = zeros(size(gt, 1), 1);
            for k = 1:size(gt, 1)
                ious(k) = calculate_iou(result.res(k,:), gt(k,:));
            end
            avg_iou = mean(ious);
            success_rate = sum(ious > 0.5) / length(ious);
            
            fprintf('    Average IoU: %.3f\n', avg_iou);
            fprintf('    Success Rate (IoU>0.5): %.3f\n', success_rate);
        end
        
    catch err
        fprintf('  ERROR: %s\n', err.message);
    end
end

fprintf('\n========================================\n');
fprintf('Test completed\n');
fprintf('========================================\n');

function iou = calculate_iou(box1, box2)
    % box format: [x, y, width, height]
    x1 = max(box1(1), box2(1));
    y1 = max(box1(2), box2(2));
    x2 = min(box1(1) + box1(3), box2(1) + box2(3));
    y2 = min(box1(2) + box1(4), box2(2) + box2(4));
    
    if x2 < x1 || y2 < y1
        iou = 0;
        return;
    end
    
    intersection = (x2 - x1) * (y2 - y1);
    area1 = box1(3) * box1(4);
    area2 = box2(3) * box2(4);
    union = area1 + area2 - intersection;
    
    iou = intersection / union;
end
