import pandas as pd
import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange, traffic_dataset
from libs.event_list import physical_adj, traffic_flow_data, get_event_list
from tqdm import tqdm
from torch.utils.data import DataLoader


def get_dataloaders_from_index_data(
        data_dir, tod=False, dow=False, dom=False, batch_size=64, log=None

):

    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    features = [0]
    if tod:  # time of day
        features.append(1)
    if dow:  # day of week
        features.append(2)

    data = data[..., features]
    num_nodes = data.shape[1]

    index = np.load(os.path.join(data_dir, "index.npz"))
    adj = np.load(os.path.join(data_dir,'adj.npy'))
    adj[np.isnan(adj)] = 0.
    adj[np.isinf(adj)] = 0.

    event_flag = 0.01  # 0.1
    event_list = get_event_list(adj, event_flag)

    train_index = index["train"]
    val_index = index["val"]

    finally_index = np.where(index["test"][:, -1] == event_list.shape[0])[0].item()
    test_index = index["test"][:finally_index+1]

    train_index_last = train_index[-1]
    val_index_last = val_index[-1]
    test_index_last = test_index[-1]

    train_flow_data = data[:train_index_last[-1]]
    A_train = adj[:train_index_last[-1]]
    train_event_data = event_list[:train_index_last[-1]]

    train_flow_dataset = []
    train_flow_label = []

    train_adj_dataset = []
    train_adj_label = []

    train_event_dataset = []

    for i in tqdm(range(0, train_flow_data.shape[0])):

        if (i + 24) < train_flow_data.shape[0]:
            train_flow_dataset.append(
                train_flow_data[i:i + 12])
            train_flow_label.append(train_flow_data[i + 12:i + 24][..., 0])

            train_adj_dataset.append(A_train[i + 11])
            train_adj_label.append(A_train[i + 23])

            train_event_dataset.append(train_event_data[i:i + 12])

    val_flow_data = data[train_index_last[-1]:val_index_last[-1]]
    A_val = adj[train_index_last[-1]:val_index_last[-1]]
    val_event_data = event_list[train_index_last[-1]:val_index_last[-1]]

    val_flow_dataset = []
    val_flow_label = []

    val_adj_dataset = []
    val_adj_label = []

    val_event_dataset = []

    for i in tqdm(range(0, val_flow_data.shape[0])):
        if (i + 24) < val_flow_data.shape[0]:
            val_flow_dataset.append(val_flow_data[i:i + 12])
            val_flow_label.append(val_flow_data[i + 12:i + 24][..., 0])

            val_adj_dataset.append(A_val[i + 11])
            val_adj_label.append(A_val[i + 23])

            val_event_dataset.append(val_event_data[i:i + 12])

    test_flow_data = data[val_index_last[-1]:test_index_last[-1]]
    A_test = adj[val_index_last[-1]:test_index_last[-1]]
    test_event_data = event_list[val_index_last[-1]:test_index_last[-1]]

    test_flow_dataset = []
    test_flow_label = []

    test_adj_dataset = []
    test_adj_label = []

    test_event_dataset = []

    for i in tqdm(range(0, test_flow_data.shape[0])):
        if (i + 24) < test_flow_data.shape[0]:
            test_flow_dataset.append(test_flow_data[i:i + 12])
            test_flow_label.append(test_flow_data[i + 12:i + 24][..., 0])

            test_adj_dataset.append(A_test[i + 11])
            test_adj_label.append(A_test[i + 23])

            test_event_dataset.append(test_event_data[i:i + 12])

    scaler = StandardScaler(mean=np.array(train_flow_dataset)[..., 0].mean(),
                            std=np.array(train_flow_dataset)[..., 0].std())


    train_dataset = traffic_dataset(train_flow_dataset, train_flow_label,
                                    train_adj_dataset, train_adj_label,
                                    train_event_dataset, scaler)

    val_dataset = traffic_dataset(val_flow_dataset, val_flow_label,
                                  val_adj_dataset, val_adj_label,
                                  val_event_dataset, scaler)

    test_dataset = traffic_dataset(test_flow_dataset, test_flow_label,
                                   test_adj_dataset, test_adj_label,
                                   test_event_dataset, scaler)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # false
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # false
        drop_last=True
    )
    return train_loader, val_loader, test_loader, scaler

