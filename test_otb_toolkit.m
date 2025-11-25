% Test ECR_CSRT tracker on OTB benchmark
% This will auto-download sequences if needed

cd('e:\SourceCode\C2P\Project\ECR_Tracking\toolkit');

% Initialize
startup;

% Load configurations
trackers = config_trackers;
sequences = config_sequences;

% Filter to only ECR_CSRT tracker
ecr_csrt = [];
for i = 1:length(trackers)
    if strcmp(trackers{i}.name, 'ECR_CSRT')
        ecr_csrt = trackers(i);
        break;
    end
end

if isempty(ecr_csrt)
    error('ECR_CSRT tracker not found');
end

fprintf('Found ECR_CSRT tracker\n');
fprintf('Found %d sequences\n', length(sequences));

% Test with first 3 sequences only
test_seqs = sequences(1:min(3, length(sequences)));

fprintf('Testing on sequences:\n');
for i = 1:length(test_seqs)
    fprintf('  %d. %s\n', i, test_seqs{i}.name);
end

% Run OPE evaluation
fprintf('\n=== Running OPE Evaluation ===\n');
OPE_evaluate(test_seqs, ecr_csrt);

fprintf('\nEvaluation completed!\n');
fprintf('Check results in: toolkit/results/OPE/\n');
