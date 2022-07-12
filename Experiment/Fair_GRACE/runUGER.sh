for method in uge-r uge-w uge-c
do
    for attr in age gender occupation
    do
        for rw in 0.2 2 20 40
        do
            for drop in 0.2
            do
                python3 run_fairGrace.py --epochs 200 --dataname 'movielens' --debias_method="uge-r" --debias_attr $attr --reg_weight $rw --der1 $drop --der2 $drop --dfr1 $drop --dfr2 $drop --debias_method $method
            done
        done
    done
done