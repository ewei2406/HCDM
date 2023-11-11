for method in uge-r 
do
    for attr in age gender occupation
    do
        for rw in 0.2 2 20 40
        do
            for drop in 0.2
            do
                for seed in 100 101
                do
                    python3 run_fairGrace.py --epochs 200 --dataname 'movielens' --debias_attr $attr --reg_weight $rw --der1 $drop --der2 $drop --dfr1 $drop --dfr2 $drop --debias_method $method --seed $seed
                done
            done
        done
    done
done
