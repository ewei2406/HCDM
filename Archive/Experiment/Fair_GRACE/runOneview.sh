for seed in 100 101 102 103 104 105 106 107 108 109
do
  for one_view in Y N
  do
    for dr in 0.15 0.2 0.3
    do
      python3 main.py --one_view $one_view --seed $seed --der1 $dr --der2 $dr --dfr1 $dr --dfr2 $dr
    done
  done
done