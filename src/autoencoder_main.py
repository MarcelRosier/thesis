import argparse

from autoencoder import main

# arg setup
parser = argparse.ArgumentParser()
parser.add_argument(
    '--cuda_id', '-c', help='specify the id of the to be used cuda device', type=int, default=0)
args = parser.parse_args()
# print(args.cuda_id)

# run
main.run(cuda_id=args.cuda_id)
