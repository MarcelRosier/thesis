import argparse

from torch.utils.data import DataLoader, dataset

from autoencoder import main
from autoencoder.dataset import TumorT1CDataset

# arg setup
parser = argparse.ArgumentParser()
parser.add_argument(
    '--cuda_id', '-c', help='specify the id of the to be used cuda device', type=int, default=0)
args = parser.parse_args()
# print(args.cuda_id)

# run
main.run(cuda_id=args.cuda_id)
