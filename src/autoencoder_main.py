from torch.utils.data import dataset
from autoencoder.dataset import TumorT1CDataset
from torch.utils.data import DataLoader


# run
dataset = TumorT1CDataset()


train_loader = DataLoader(dataset=dataset,
                          batch_size=50,
                          shuffle=True,
                          num_workers=2)

for i, (inputs, labels) in enumerate(train_loader):
    print(f'batch_step= {i + 1} / { len(train_loader)}')
