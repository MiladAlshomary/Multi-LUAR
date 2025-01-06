#!/bin/bash
#
#SBATCH --job-name=ta2
#SBATCH --ntasks=8
#SBATCH --time=920:00
#SBATCH --nodelist=nlpgpu04

#SBATCH --partition=p_nlp
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=500GB

cd /nlp/data/taing/pausit-eval-ta2

# Function to run the job with haystack sizes
run_job() {
    input_dir="$1"
    size_id="$2"

    srun python3 main.py --input-dir "$input_dir" --output-dir output --ground-truth-dir "$input_dir" --run-id "$size_id" --query-identifier authorIDs --candidate-identifier authorSetIDs -s -ta1 l2v_g2v -g -ta2 knn > "output_${size_id}.log" 2>&1 &
}

# Launch jobs in parallel
run_job "batch_data/50" 50
wait
run_job "batch_data/100" 100
wait
run_job "batch_data/500" 500
wait
run_job "batch_data/1000" 1000
wait
# run_job "batch_data/5000" 5000
# run_job "batch_data/10000" 10000
# run_job "batch_data/15000" 15000
# run_job "batch_data/21000" 21000
