runtime_folder=$(date +%Y%m%d_%H%M%S)
path=experiment_path/${setting}_${freqs}/flow_matching/${runtime_folder}
mkdir -p $path

python plate_optim/guided_flow.py \
        --dir $path \
        --regression_path $regression_path_noise \
        --flow_matching_path $flow_matching_path \
        --batch_size 16 \
        --alpha 1 \
        --n_plates 1312 \
        --n_candidates 4 \
        --extra_conditions $extra_conditions \
        --min_freq $min_freq \
        --max_freq $max_freq \
        --n_freqs $n_freqs \
        --step_size 0.05 \
        --logging True \
        --return_intermediates True \
        --run_name flow_matching${setting}_${freqs}