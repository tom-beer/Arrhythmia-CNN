from __future__ import print_function
import torch
import torch.utils.data
import numpy as np
import pandas as pd
import wfdb
import os

from torch import nn, optim
from torch.utils.data.dataset import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

is_cuda = False
num_epochs = 100
batch_size = 10
torch.manual_seed(46)
log_interval = 10
in_channels_ = 1
num_segments_in_record = 100
segment_len = 3600
num_records = 48
num_classes = 12
allow_label_leakage = True

device = torch.device("cuda:2" if is_cuda else "cpu")
index_set = (num_records * num_segments_in_record if allow_label_leakage else num_records)
train_ids, test_ids = train_test_split(np.arange(index_set), train_size=.8, random_state=46)
scaler = MinMaxScaler(feature_range=(0, 1), copy=False)


class CustomDatasetFromCSV(Dataset):
    def __init__(self, data_path, transforms_=None):
        self.df = pd.read_pickle(data_path)
        self.transforms = transforms_

    def __getitem__(self, index):

        row = self.df.iloc[index]
        signal = row['signal']
        target = row['target']
        if self.transforms is not None:
            signal = self.transforms(signal)

        return signal, target

    def __len__(self):
        return self.df.shape[0]


train_dataset = CustomDatasetFromCSV('./data/sample_df_bar.pkl')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = CustomDatasetFromCSV('./data/sample_df_bar.pkl')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def GenerateLR(max_lr, min_lr_ratio, anneal_cycle_pct, tot_num_iter):
    min_lr = max_lr / min_lr_ratio
    half_cyc_len = int((np.floor(100 - anneal_cycle_pct) / 100) * tot_num_iter / 2)
    anneal_len = int(tot_num_iter - 2 * half_cyc_len)
    upVec = np.linspace(min_lr, max_lr, half_cyc_len)
    downVec = np.flip(upVec)
    annealVec = np.flip(np.linspace(min_lr / 100, min_lr, anneal_len))
    cyclic_lr = (upVec.tolist()) + (downVec.tolist()) + (annealVec.tolist())
    return cyclic_lr


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


def basic_layer(in_channels, out_channels, kernel_size, batch_norm=False, max_pool=True, conv_stride=1, padding=0
                , pool_stride=2, pool_size=2):
    layer = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=conv_stride,
                  padding=padding),
        nn.ReLU())
    if batch_norm:
        layer = nn.Sequential(
            layer,
            nn.BatchNorm1d(num_features=out_channels))
    if max_pool:
        layer = nn.Sequential(
            layer,
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride))

    return layer


class arrhythmia_classifier(nn.Module):
    def __init__(self, in_channels=in_channels_):
        super(arrhythmia_classifier, self).__init__()
        self.cnn = nn.Sequential(
            basic_layer(in_channels=in_channels, out_channels=128, kernel_size=50, batch_norm=True, max_pool=True,
                        conv_stride=3, pool_stride=3),
            basic_layer(in_channels=128, out_channels=32, kernel_size=7, batch_norm=True, max_pool=True,
                        conv_stride=1, pool_stride=2),
            basic_layer(in_channels=32, out_channels=32, kernel_size=10, batch_norm=False, max_pool=False,
                        conv_stride=1),
            basic_layer(in_channels=32, out_channels=128, kernel_size=5, batch_norm=False, max_pool=True,
                        conv_stride=2, pool_stride=2),
            basic_layer(in_channels=128, out_channels=256, kernel_size=15, batch_norm=False, max_pool=True,
                        conv_stride=1, pool_stride=2),
            basic_layer(in_channels=256, out_channels=512, kernel_size=5, batch_norm=False, max_pool=False,
                        conv_stride=1),
            basic_layer(in_channels=512, out_channels=128, kernel_size=3, batch_norm=False, max_pool=False,
                        conv_stride=1),
            Flatten(),
            nn.Linear(in_features=1152, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Softmax()
        )

    def forward(self, x, ex_features=None):
        # encoded = self.encode(x)
        # us_decoded = self.us_decoder(encoded)
        # ex_decoded = self.ex_decoder(ex_features)
        # aggr_rep = torch.cat([ex_decoded.to(device), us_decoded], dim=1)
        # z = self.aggr_decoder(aggr_rep)
        return self.cnn(x)


def calc_next_len_conv1d(current_len=112500, kernel_size=16, stride=8, padding=0, dilation=1):
    return int(np.floor((current_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


model = arrhythmia_classifier().to(device).double()
lr = 0.0003
num_of_iteration = len(train_dataset) // batch_size

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
criterion = nn.NLLLoss()
cyclic_lr = GenerateLR(max_lr=lr, min_lr_ratio=10, anneal_cycle_pct=10,
                       tot_num_iter=(num_epochs * num_of_iteration))


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # if len(cyclic_lr) > 0:
        #     lr = cyclic_lr.pop(0)
        # for g in optimizer.param_groups:
        #     g['lr'] = lr
        optimizer.zero_grad()
        # features = np.random.randn(batch_size, 40)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch = model(data)
            loss = criterion(recon_batch, data.view(recon_batch.shape[0], -1))
            test_loss += loss.item()
            # import matplotlib.pyplot as plt
            # plt.plot(np.squeeze(data[0, :, :].cpu().detach().numpy()))
            # plt.plot(np.squeeze(recon_batch[0, :].cpu().detach().numpy()))

            if batch_idx == 0:
                n = min(data.size(0), 4)
                # comparison = torch.cat([data[:n],
                #                       recon_batch.view(batch_size, 1, 28, 28)[:n]])
                # save_image(comparison.cpu(),
                #          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.5f}'.format(test_loss))
    # scheduler.step(test_loss)
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')


if __name__ == "__main__":
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')
