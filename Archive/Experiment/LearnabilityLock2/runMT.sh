seed=124
gpu=1

for g0_method in random bias many_clusters large_cluster
do
  for dataset in cora citeseer flickr BlogCatalog Polblogs
  do
    for attack_method in sll_no_g noise heuristic
    do
      python3 mtt.py \
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