import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import seaborn as sns

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from data.dataset import SEEDIVDataset, AsyncBatchLoader

# Установка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

sns.set(font_scale=1.4, style="whitegrid")
figure_format = "retina"

class DCGRU(nn.Module):
    def __init__(self, input_dim, hiden_dim, K, normalization="sym"):
        super().__init__()
        self.input_dim = input_dim
        self.hiden_dim = hiden_dim
        self.K = K

        self.reset_conv = gnn.ChebConv(
            input_dim + hiden_dim, hiden_dim, K, normalization
        ).to(device)
        self.update_conv = gnn.ChebConv(
            input_dim + hiden_dim, hiden_dim, K, normalization
        ).to(device)
        self.mem_conv = gnn.ChebConv(
            input_dim + hiden_dim, hiden_dim, K, normalization
        ).to(device)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, edge_idx, edge_attr):
        hiden = torch.zeros(x[0].shape[0], self.hiden_dim).to(device)
        hiden_arr = []

        for x_item, edge_item, attr_item in zip(x, edge_idx, edge_attr):
            x_item = x_item.to(device)
            edge_item = edge_item.to(device)
            attr_item = attr_item.to(device)

            combined = torch.cat((x_item, hiden), dim=1)

            r = self.sigmoid(
                self.reset_conv(combined, edge_item, attr_item)
            )
            u = self.sigmoid(
                self.update_conv(combined, edge_item, attr_item)
            )
            c = self.tanh(
                self.mem_conv(
                    torch.cat((x_item, r * hiden), dim=1), edge_item, attr_item
                )
            )

            hiden = u * hiden + (1 - u) * c
            hiden_arr.append(hiden)

        return hiden, hiden_arr


class EmoClassifier(nn.Module):
    def __init__(self, num_classes, num_nodes, input_dim, hiden_dim, K, normalization="sym"):
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = DCGRU(input_dim, 128, K, normalization)
        self.encoder2 = DCGRU(128, 128, K, normalization)
        self.decoder = DCGRU(128, 128, K, normalization)

        self.fc = nn.Linear(128 * 62, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.9)
        self.drp = nn.Dropout(0.9)
        self.relu = nn.ReLU()

    def forward(self, x, edge_idx, edge_attr):
        h1, s = self.encoder(x, edge_idx, edge_attr)
        h2, _ = self.encoder2(s, edge_idx, edge_attr)

        for _ in range(1):
            h1, _ = self.decoder([h2], edge_idx, edge_attr)

        last_out = torch.stack(torch.split(h1, self.num_nodes, dim=0), dim=0)
        last_out = torch.flatten(last_out, 1)

        logits = self.fc2(self.drp(self.fc(self.relu(self.dropout(last_out)))))

        # max-pooling over nodes
        # pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)

        return logits

