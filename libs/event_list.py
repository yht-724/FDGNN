from scipy.stats import entropy
import numpy as np
import pandas as pd
from .utils import StandardScaler
import torch.nn.functional as F
import torch
from tqdm import tqdm
import os


def physical_adj(distancePath,num_nodes):
    data = pd.read_csv(distancePath).to_numpy()
    dist = np.zeros((num_nodes, num_nodes))
    for i in range(data.shape[0]):
        dist[int(data[i][0]), int(data[i][1])] = 1. / data[i][2]
    Max = dist.max()
    Min = dist.min()
    dist = (dist - Min) / (Max - Min)

    return dist


def traffic_flow_data(data_path):
    data = np.load(data_path)['data'][:, :, 0]
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    scaler = StandardScaler(mean=mean, std=std)
    data = scaler.transform(data)
    return data


def semantic_adj(data_path):
    data = traffic_flow_data(data_path)
    A_list = []

    for i in tqdm(range(11, data.shape[0])):
        if (i + 12) < data.shape[0]:
            A = np.zeros((307, 307))
            for idx, x in np.ndenumerate(A):
                A[idx[0], idx[1]] = entropy(data[i - 11:i + 1, idx[0]], data[i - 11:i + 1, idx[1]])

            Min = A.min()
            Max = A.max()
            if (Max - Min) != 0:
                A = (A - Min) / (Max - Min)
            A_list.append(A)
    np.save('../data/PEMS04/adj.npy', np.array(A_list))


def get_event_list(all_A, event_flag):
    event_list = []
    for i, j in zip(all_A[:-1], all_A[1:]):
        event = j - i
        event[event >= event_flag] = 1
        event[event <= -event_flag] = 2
        event[(event < event_flag) & (event > -event_flag)] = 0
        event_list.append(event)
    event_list = np.array(event_list)
    return event_list


if __name__ == '__main__':
    data_path = '../data/PEMS04/data.npz'
    semantic_adj(data_path)
