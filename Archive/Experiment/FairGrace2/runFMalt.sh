reg_weight=10
ratios=(5 10 20)
ratio=10
r_methods=(aug-r opposite-r r)
n_methods=(none random)
attrs=(0 1 2)

for ratio in ${ratios[@]}
do
  for attr in ${attrs[@]}
  do
    for method in ${r_methods[@]}
    do
      python3 FM.py --reg_weight $reg_weight --debias_method $method --sim_diff_ratio $ratio --seed 100 --debias_attr $attr --epochs 200 --gpu 2
    done
  done
done


