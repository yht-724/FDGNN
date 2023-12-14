import numpy as np
import os
from torch.utils.data import DataLoader
from .utils import Data_Batch, generate_event_pickle, open_pkl_file, padding_full
from .event_list import get_event_list, semantic_adj, physical_adj


def generate_hawkes_data(data_dir, num_nodes):
    print('Loading hawkes event data...')

    all_data = np.load(os.path.join(data_dir, "data.npz"))['data'][:, :, 0]

    if os.path.exists(os.path.join(data_dir, "adj.npy")) == False:
        semantic_adj(os.path.join(data_dir, "data.npz"))
        A = np.load(os.path.join(data_dir, "adj.npy"))
    else:
        A = np.load(os.path.join(data_dir, "adj.npy"))
    A[np.isnan(A)] = 0.
    A[np.isinf(A)] = 1.

    event_flag = 0.01
    event_list = get_event_list(A, event_flag)

    all_event_data = event_list

    dist = physical_adj(os.path.join(data_dir, "distance.csv"), num_nodes)
    non_zero_index = np.nonzero(dist)
    p1, p2 = non_zero_index

    if os.path.exists(os.path.join(data_dir, "all.pkl")) == False:
        generate_event_pickle('all.pkl', all_event_data, p1, p2, data_dir)

    all_time_duration, type_all, seq_lens_all, all_event_timeIndex, node_ij_all = open_pkl_file(
        os.path.join(data_dir, "all.pkl"), 'all')

    type_size = 2
    all_time_duration, type_all = padding_full(all_time_duration, type_all, seq_lens_all, type_size)

    all_event_dataset = Data_Batch(all_time_duration, type_all, seq_lens_all)

    event_batch = 64
    all_event_loader = DataLoader(all_event_dataset, batch_size=event_batch, shuffle=False)

    print('hawkes event data is done')

    return all_event_loader, dist
