datasets=(cora citeseer Polblogs flickr BlogCatalog)


for dataset in "${datasets[@]}"
do
    for seed in {100..104}
    do

        python3 SelectiveAttack.py --dataset $dataset --seed $seed --check_universal Y

    done
done

seeds=()