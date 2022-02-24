import argparse

# arg setup
parser = argparse.ArgumentParser()
parser.add_argument(
    '--cuda_id', '-c', help='specify the id of the to be used cuda device', type=int, default=0)
parser.add_argument(
    '--mode', '-m', help='select mode: 1= train, 2=create test data set, 3= run encoded similarity check, 4=query', type=int, default=0)
args = parser.parse_args()
# print(args.cuda_id)

# run
if args.mode == 1:
    print(f"Run training main on {args.cuda_id=}")
    from autoencoder import train
    train.run(cuda_id=args.cuda_id)
elif args.mode == 2:
    print(f"Run enc gen main on {args.cuda_id=}")
    from autoencoder import gen_encoded
    gen_encoded.run(cuda_id=args.cuda_id)
    pass
elif args.mode == 3:
    print(f"Run similarity gen main on {args.cuda_id=}")
    from autoencoder import encoded_similarity_check
    encoded_similarity_check.run(
        real_tumor="tgm001_preop")
elif args.mode == 4:
    from autoencoder import query
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)
    device = torch.device(
        f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    query.run(processes=32)
