seed=124
gpu=0

for g0_method in random bias many_clusters large_cluster
do
  for dataset in cora citeseer Polblogs BlogCatalog flickr
  do
    # for attack_method in sll sll_no_g noise heuristic
    # do
    python3 compare.py \
    --dataset $dataset \
    --g0_method $g0_method \
    --seed $seed \
    --gpu_id $gpu \
    --save_results Y \
    --save_graph N
    # done
  done
done