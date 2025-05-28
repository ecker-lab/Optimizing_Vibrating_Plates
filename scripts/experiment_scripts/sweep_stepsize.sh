base_path='path'
mkdir -p $base_path

step_values=(0.01 0.025 0.05 0.1)

source scripts/setting_a_args.sh
source scripts/range100_200.sh

for step_size in "${step_values[@]}"; do
    path=$base_path/flow_matching_${step_size}
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
        --step_size $step_size \
        --logging True \
        --return_intermediates False \
        --run_name step_size_sweep_${step_size}_

    source scripts/fem_call.sh 

done

