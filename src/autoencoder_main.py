import argparse

# arg setup
parser = argparse.ArgumentParser()
parser.add_argument(
    '--cuda_id', '-c', help='specify the id of the to be used cuda device', type=int, default=0)
parser.add_argument(
    '--mode', '-m', help='select mode: 1= train, 2=eval, 3= X', type=int, default=0)
args = parser.parse_args()
# print(args.cuda_id)

# run
if args.mode == 1:
    print(f"Run training main on {args.cuda_id=}")
    from autoencoder import train
    train.run(cuda_id=args.cuda_id)
elif args.mode == 2:
    print(f"Run testing main on {args.cuda_id=}")
    from autoencoder import test
    test.run(cuda_id=args.cuda_id)
    pass
elif args.mode == 3:
    from autoencoder import encoded_similarity_check
    encoded_similarity_check.run(
        real_tumor="tgm001_preop", syn_subset=(3000, 3200))
    pass
