datasets=(cora citeseer flickr BlogCatalog)


for dataset in "${datasets[@]}"
do
    for seed in {100..101}
    do
        for ptb_rate in 0.25 0.5
        do
            python3 Universal.py --ptb_rate $ptb_rate --dataset $dataset --seed $seed --save_location "./RandResults.csv" --save Y --top N
        done
    done
done

seeds=()