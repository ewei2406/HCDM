import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--method', type=str, default="A", help='Drop feature ratio of the 2nd augmentation.')

args = parser.parse_args()

print(args)