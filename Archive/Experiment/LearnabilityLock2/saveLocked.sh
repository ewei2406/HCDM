seed=123
gpu=0

for g0_method in random
do
  for dataset in cora citeseer Polblogs BlogCatalog flickr
  do
    for attack_method in sll sll_no_g noise heuristic
    do
      python3 main.py \
      --dataset $dataset \
      --g0_method $g0_method \
      --attack_method $attack_method \
      --seed $seed \
      --gpu $gpu \
      --save_results Y \
      --save_graph Y
    done
  done
done