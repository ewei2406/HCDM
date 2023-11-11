for method in uge-w
do
    for attr in age gender occupation
    do
        for drop in 0.2
        do
            python3 UGEW.py --epochs 200 --dataname 'movielens' --debias_attr $attr  --der1 $drop --der2 $drop --dfr1 $drop --dfr2 $drop --debias_method $method --reg_weight 0.2
        done
    done
done