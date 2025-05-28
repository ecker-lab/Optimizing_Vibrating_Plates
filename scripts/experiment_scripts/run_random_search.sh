runtime_folder=$(date +%Y%m%d_%H%M%S)
path=experiment_path/${setting}_${freqs}/random_search/${runtime_folder}
mkdir -p $path

python plate_optim/random_search.py \
        --dir $path \
        --regression_path $regression_path_no_noise \
        --batch_size 32 \
        --extra_conditions $extra_conditions \
        --min_freq $min_freq \
        --max_freq $max_freq \
        --n_freqs $n_freqs \
        --n_plates 40000 \
        --n_candidates 4 \
        --logging True \
        --name random_search_${setting}_${freqs}
