#!/bin/bash
# --- this job will be run on any available node
# and simply output the node's hostname to
# my_job.output
#SBATCH --partition=gpu
#SBATCH --job-name="RunFM"
#SBATCH --error="my_job.err"
#SBATCH --output="RunFM.output"

for method in uge-r none
do
  for reg_weight in 0.2 2 20 40
  do
    for ratio in 0.5 2 4 10
    do
      for attr in 0 1 2
      do
        python3 RunFM.py --reg_weight $reg_weight --debias_method $method --enable_heuristic Y --sim_diff_ratio $ratio --seed 100 --debias_attr $attr
      done
    done
  done
done

for reg_weight in 0.2 2 20 40
do
  for attr in 0 1 2
  do
    python3 RunFM.py --reg_weight $reg_weight --debias_method uge-r --enable_heuristic N --sim_diff_ratio 3 --seed 100 --debias_attr $attr
  done
done

for attr in 0 1 2
do
  python3 RunFM.py --reg_weight 1 --debias_method none --enable_heuristic N --sim_diff_ratio 3 --seed 100 --debias_attr $attr
done