python plate_optim/regression/run.py\
    --config regression/2_5k15.yaml\
    --batch_size 64\
    --scaling_factor 32\
    --add_noise 0.75\
    --dir data_reduction/regression_2_5k15

python plate_optim/regression/run.py\
    --config regression/5k15.yaml\
    --batch_size 64\
    --scaling_factor 32\
    --add_noise 0.75\
    --dir data_reduction/regression_5k15

python plate_optim/regression/run.py\
    --config regression/25k15.yaml\
    --batch_size 64\
    --scaling_factor 32\
    --add_noise 0.75\
    --dir data_reduction/regression_25k15