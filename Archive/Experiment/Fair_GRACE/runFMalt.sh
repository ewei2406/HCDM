for method in uge-c
do
    for attr in occupation gender age
    do
        for drop in 0.2
        do
            for rw in 0.2 2 20 40
            do
                python3 UGEW.py --epochs 200 --dataname 'movielens' --debias_attr $attr  --der1 $drop --der2 $drop --dfr1 $drop --dfr2 $drop --debias_method $method --reg_weight $rw --gpu 0
            done
        done
    done
done