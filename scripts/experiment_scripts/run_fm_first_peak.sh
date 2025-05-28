path=experiment_path/first_peak_${setting}/
mkdir -p $path

python plate_optim/guided_flow.py \
        --dir $path \
        --regression_path $regression_path_noise \
        --flow_matching_path $flow_matching_path \
        --batch_size 8 \
        --alpha 1 \
        --n_plates 640 \
        --n_candidates 4 \
        --extra_conditions $extra_conditions \
        --min_freq 1 \
        --max_freq 300 \
        --n_freqs 300 \
        --logging True \
        --return_intermediates True \
        --move_first_peak True \
        --run_name first_peak 

