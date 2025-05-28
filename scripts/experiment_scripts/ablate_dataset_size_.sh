source scripts/setting_a_args.sh
source scripts/range200_250.sh
values=("2.5" "5" "10" "25" '50')
# values=('10')

# Loop through each value
for val in "${values[@]}"; do
    source "scripts/regression/${val}k15ckpt.sh" # get ckpt path
    path="experiment_path/${setting}_${freqs}/${val}/"
    mkdir -p $path

    # Run the Python script
    python plate_optim/guided_flow.py \
        --dir "$path" \
        --regression_path "$regression_path_noise" \
        --flow_matching_path "$flow_matching_path" \
        --batch_size 16 \
        --alpha 1 \
        --n_plates 640 \
        --n_candidates 12 \
        --extra_conditions "$extra_conditions" \
        --min_freq "$min_freq" \
        --max_freq "$max_freq" \
        --n_freqs "$n_freqs" \
        --step_size 0.05 \
        --logging True \
        --return_intermediates True \
        --run_name "${val}${setting}_${freqs}"
    source scripts/fem_call.sh 

done

