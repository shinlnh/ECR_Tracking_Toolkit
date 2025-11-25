function attr_auc = compute_attr_auc()
% Compute AUC theo từng thuộc tính OTB cho các tracker đã có sẵn kết quả.
% Dùng các đường cong success đã lưu trong perfmat/OPE/perfplot_curves_OPE.mat
% và nhãn thuộc tính trong tracker_benchmark_v1.0/anno/att.

toolkit_path = fileparts(mfilename('fullpath'));

perf_file = fullfile(toolkit_path, 'perfmat', 'OPE', 'perfplot_curves_OPE.mat');
if ~exist(perf_file, 'file')
    error('Không tìm thấy perfmat: %s', perf_file);
end
load(perf_file, 'success_curve', 'nameTrkAll');

seq_file = fullfile(toolkit_path, 'sequences', 'SEQUENCES');
seq_list = strtrim(splitlines(fileread(seq_file)));
seq_list = seq_list(~cellfun(@isempty, seq_list));

att_dir = fullfile(toolkit_path, 'tracker_benchmark_v1.0', 'anno', 'att');
att_names = { ...
    'illumination variation', ...
    'out-of-plane rotation', ...
    'scale variation', ...
    'occlusion', ...
    'deformation', ...
    'motion blur', ...
    'fast motion', ...
    'in-plane rotation', ...
    'out of view', ...
    'background clutter', ...
    'low resolution' ...
    };

n_attr = numel(att_names);
n_trk = numel(nameTrkAll);

attr_auc = nan(n_attr, n_trk);
att_flags = nan(numel(seq_list), n_attr);
has_attr = false(numel(seq_list), 1);

% Đọc nhãn thuộc tính (0/1) cho từng chuỗi nếu có file.
for s = 1:numel(seq_list)
    att_file = fullfile(att_dir, [lower(seq_list{s}) '.txt']);
    if exist(att_file, 'file')
        flags = readmatrix(att_file);
        has_attr(s) = true;
        att_flags(s, :) = flags(:).';
    end
end

% Tính AUC cho từng thuộc tính.
for a = 1:n_attr
    seq_idx = find(att_flags(:, a) == 1);
    if isempty(seq_idx)
        continue;
    end
    for t = 1:n_trk
        auc_seq = cellfun(@(c) mean(c), success_curve(t, seq_idx));
        attr_auc(a, t) = mean(auc_seq);
    end
end

% Xuất bảng CSV để tiện xem/sắp xếp.
tracker_names = nameTrkAll(:).';
tbl = array2table(attr_auc, 'VariableNames', matlab.lang.makeValidName(tracker_names));
tbl.Attribute = att_names.';
tbl = movevars(tbl, 'Attribute', 'Before', 1);
csv_path = fullfile(toolkit_path, 'figs', 'OPE', 'attribute_auc.csv');
writetable(tbl, csv_path);

% Vẽ heatmap AUC (ẩn figure để chạy batch).
fig = figure('Visible', 'off', 'Position', [100 100 1200 520]);
h = heatmap(tracker_names, att_names, attr_auc, 'Colormap', parula);
h.Title = 'OTB attribute AUC (success, OPE)';
h.XLabel = 'Tracker';
h.YLabel = 'Attribute';
h.CellLabelFormat = '%.3f';
png_path = fullfile(toolkit_path, 'figs', 'OPE', 'attribute_auc_heatmap.png');
exportgraphics(fig, png_path, 'Resolution', 200);
close(fig);

% In nhanh top-3 tracker cho từng thuộc tính ra console khi chạy batch.
for a = 1:n_attr
    seq_idx = find(att_flags(:, a) == 1);
    if isempty(seq_idx)
        fprintf('%s: không có chuỗi được gán thuộc tính.\n', att_names{a});
        continue;
    end
    [scores, order] = sort(attr_auc(a, :), 'descend');
    topk = min(3, numel(order));
    fprintf('%s (|seq|=%d) top-%d: ', att_names{a}, numel(seq_idx), topk);
    for k = 1:topk
        fprintf('%s %.3f ', tracker_names{order(k)}, scores(k));
    end
    fprintf('\n');
end
end
