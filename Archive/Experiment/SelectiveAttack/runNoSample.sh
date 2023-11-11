datasets=(cora citeseer Polblogs flickr BlogCatalog)


for dataset in "${datasets[@]}"
do
    for seed in {100..104}
    do
        for ptb_rate in 0.1 0.25 0.5
        do
            python3 SelectiveAttack.py --ptb_rate $ptb_rate --dataset $dataset --seed $seed --do_sampling N --save Y --save_location "./NoSampleResults.csv" --check_universal Y
        done
    done
done

seeds=()