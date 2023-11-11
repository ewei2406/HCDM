gpu=1
epochs=5

for seed in 100 101 102 103 104
do
  for method in regular oneview
  do
    python3 allMethods.py --method $method --gpu $gpu --epochs $epochs
  done

  for method in grad_positive grad_negative oneview_grad_positive oneview_grad_negative
  do
    for grad_target in 0.005 0.01 0.05 0.1
    do
      python3 allMethods.py --method $method --grad_target $grad_target --gpu $gpu --epochs $epochs
    done
  done
done