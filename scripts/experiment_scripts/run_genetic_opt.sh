runtime_folder=$(date +%Y%m%d_%H%M%S)
path=experiment_path/${setting}_${freqs}/genetic_opt_medium_run/${runtime_folder}
mkdir -p $path

python plate_optim/genetic_opt.py \
        --dir $path \
        --regression_path $regression_path_no_noise \
        --extra_conditions $extra_conditions \
        --min_freq $min_freq \
        --max_freq $max_freq \
        --n_freqs $n_freqs \
        --n_candidates 4 \
        --max_n_arcs 2 \
        --max_n_lines 2 \
        --max_n_quads 2 \
        --n_pop 10 \
        --n_max_iter 100 \
        --vectorized True \
        --batch_size 72


