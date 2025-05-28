base_path=experiment_path/ablation_postprocessing
mkdir -p $base_path


source scripts/setting_a_args.sh
source scripts/range100_200.sh

path=$base_path/False
mkdir -p $path
python plate_optim/guided_flow.py \
    --dir $path \
    --regression_path $regression_path_noise \
    --flow_matching_path $flow_matching_path \
    --batch_size 16 \
    --alpha 1 \
    --n_plates 160 \
    --n_candidates 4 \
    --extra_conditions $extra_conditions \
    --min_freq $min_freq \
    --max_freq $max_freq \
    --n_freqs $n_freqs \
    --apply_postprocessing False \
    --logging True \
    --return_intermediates False \
    --run_name postprocessing_False_

source scripts/fem_call.sh 



path=$base_path/True
mkdir -p $path
python plate_optim/guided_flow.py \
    --dir $path \
    --regression_path $regression_path_noise \
    --flow_matching_path $flow_matching_path \
    --batch_size 16 \
    --alpha 1 \
    --n_plates 160 \
    --n_candidates 4 \
    --extra_conditions $extra_conditions \
    --min_freq $min_freq \
    --max_freq $max_freq \
    --n_freqs $n_freqs \
    --apply_postprocessing True \
    --logging True \
    --return_intermediates False \
    --run_name postprocessing_True_

source scripts/fem_call.sh 
