datasets=(flickr cora citeseer Polblogs BlogCatalog)
gpu=1

for dataset in "${datasets[@]}"
do

  for method in simplistic1 simplistic2
  do
  python3 AllMethods.py \
    --ptb_rate 0.1 \
    --dataset $dataset \
    --method $method \
    --gpu $gpu
  done

  for ptb_rate in 0.1 0.25 0.5
  do
    for method in SLL SLLnoSample noise
    do
    python3 AllMethods.py \
      --ptb_rate $ptb_rate \
      --dataset $dataset \
      --method $method \
      --gpu $gpu
    done
  done

done