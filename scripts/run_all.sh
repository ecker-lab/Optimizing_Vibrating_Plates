source scripts/setting_a_args.sh
source scripts/range100_200.sh
for i in {1..3}; do
    source scripts/experiment_scripts/run_genetic_opt.sh 
    source scripts/experiment_scripts/run_flow_matching.sh 
    source scripts/experiment_scripts/run_random_search.sh
done
