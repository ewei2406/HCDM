seed=123
gpu=1

for g0_method in random many_clusters 
do
    for dataset in BlogCatalog cora chameleon
    do
        for attack_method in noise
        do
            for switch in Y N
            do
                for nolabeltask in Y N
                do
                    python3 tester.py \
                    --dataset $dataset \
                    --attack_method $attack_method \
                    --g0_method $g0_method \
                    --seed $seed \
                    --gpu_id $gpu \
                    --switch $switch \
                    --nolabeltask $nolabeltask
                done
            done
        done
    done
done