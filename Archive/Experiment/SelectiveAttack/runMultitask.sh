datasets=(cora citeseer Polblogs BlogCatalog flickr)
gpu=1

for dataset in "${datasets[@]}"
do

  for ptb_rate in 0.1 0.25 0.5
  do
    for method in SLL
    do
    python3 Multitask.py \
      --ptb_rate $ptb_rate \
      --dataset $dataset \
      --method $method \
      --gpu $gpu
    done
  done

done