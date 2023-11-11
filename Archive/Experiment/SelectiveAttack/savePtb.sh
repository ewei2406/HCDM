datasets=(flickr BlogCatalog)


for dataset in "${datasets[@]}"
do
    python3 SelectiveAttack.py --dataset $dataset --ptb_rate 0.5 --save_perturbations Y
done