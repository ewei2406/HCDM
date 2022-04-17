datasets=(cora citeseer Polblogs flickr BlogCatalog)


for dataset in "${datasets[@]}"
do
    for seed in {100..104}
    do
        for ptb_rate in 0.25 0.5
        do
            python3 Universal.py --ptb_rate $ptb_rate --dataset $dataset --seed $seed --save_location "./Results.csv" --save Y
        done
    done
done

seeds=()