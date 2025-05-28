base_path='path'
mkdir -p $base_path

alpha_values=(0.1 0.5 1.0 1.5 2.0)

source scripts/setting_a_args.sh
source scripts/range100_200.sh

for alpha in "${alpha_values[@]}"; do
    path=$base_path/flow_matching_${alpha}
    mkdir -p $path
    python plate_optim/guided_flow.py \
        --dir $path \
        --regression_path $regression_path_noise \
        --flow_matching_path $flow_matching_path \
        --batch_size 16 \
        --alpha $alpha \
        --n_plates 160 \
        --n_candidates 4 \
        --extra_conditions $extra_conditions \
        --min_freq $min_freq \
        --max_freq $max_freq \
        --n_freqs $n_freqs \
        --step_size 0.05 \
        --logging True \
        --return_intermediates False \
        --run_name alpha_sweep_${alpha}_

    source scripts/fem_call.sh 

done

