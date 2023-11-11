seed=123
gpu=1
epochs=5

for method in walk cluster
do
    for timesteps in 200 400 600
    do
        for subgraph_size in 16 32 64
        do
            python3 diffGraph.py \
            --method $method \
            --timesteps $timesteps \
            --subgraph_size $subgraph_size
        done
    done
done