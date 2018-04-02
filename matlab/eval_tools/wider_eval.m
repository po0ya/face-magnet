function wider_eval(pred_dir,legend_name)
addpath(genpath('./plot'));
%preprocessingc
gt_dir = './ground_truth/wider_face_val.mat';
%preprocessing
pred_list = read_pred(pred_dir,gt_dir);
norm_pred_list = norm_score(pred_list);

%evaluate on different settings
setting_name_list = {'easy_val';'medium_val';'hard_val'};
setting_class = 'setting_int_final';

%Please specify your algorithm name.

for i = 1:size(setting_name_list,1)
    fprintf('Current evaluation setting %s\n',setting_name_list{i});
    setting_name = setting_name_list{i};
    gt_dir = sprintf('./ground_truth/wider_%s.mat',setting_name);
    evaluation(norm_pred_list,gt_dir,setting_name,setting_class,legend_name);
end

fprintf('Plot pr curve under overall setting.\n');
dateset_class = 'Val';

% scenario-Int:
seting_class = 'int_final';
dir_int = sprintf('./plot/baselines/%s/setting_%s',dateset_class, seting_class);
wider_plot(setting_name_list,dir_int,seting_class,dateset_class);
