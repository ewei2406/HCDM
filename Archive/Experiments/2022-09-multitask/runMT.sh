seed=124
gpu=1

for g0_method in bias many_clusters large_cluster random
do
  for dataset in BlogCatalog Polblogs flickr
  do
    for attack_method in sll sll_no_g noise heuristic
    do
      python3 Multitask.py \
      --dataset $dataset \
      --attack_method $attack_method \
      --g0_method $g0_method \
      --seed $seed \
      --gpu_id $gpu \
      --save_results Y \
      --save_graph N
    done
  done
done