datasets=(flickr BlogCatalog Polblogs)


for dataset in "${datasets[@]}"
do
    python3 CalculateMetrics.py --dataset $dataset
done
