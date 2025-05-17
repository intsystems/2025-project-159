from torch.utils.data import Dataset
from data_api import DataAPI
import torch
import numpy as np
from torch_geometric.data import Batch

import torch.multiprocessing as mp


class SEEDIVDataset(Dataset):
    def __init__(self, path, group_shape):
        super().__init__()
        self.dataset = DataAPI(path, group_shape=group_shape)
        self.dataset_size = self.dataset.root[self.dataset.LABELS].shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        return self.dataset.get_sample(index)


class AsyncBatchLoader:
    def __init__(self, dataloader, queue_size=10):
        self.dataloader = dataloader
        self.queue_size = queue_size
        self.batch_queue = mp.Queue(maxsize=queue_size)

    def worker(self):
        for i, data in enumerate(self.dataloader):
            self.batch_queue.put(data)

    def start_workers(self, num_workers=4):
        processes = []
        for i in range(num_workers):
            process = mp.Process(target=self.worker)
            process.start()
            processes.append(process)
        return processes

    def __len__(self):
        return len(self.dataloader)

    def get_batch(self):
        return self.batch_queue.get()
