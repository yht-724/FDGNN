import numpy as np
import torch
import pickle
import random
import os
import json


class StandardScaler:


    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)


def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def set_cpu_num(cpu_num: int):
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f"Shape: {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)


def vrange(starts, stops):
    """Create ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)
        
        Lengths of each range should be equal.

    Returns:
        numpy.ndarray: 2d array for each range
        
    For example:

        >>> starts = [1, 2, 3, 4]
        >>> stops  = [4, 5, 6, 7]
        >>> vrange(starts, stops)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6]])

    Ref: https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
    """
    stops = np.asarray(stops)
    l = stops - starts  # Lengths of each range. Should be equal, e.g. [12, 12, 12, ...]
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])


def print_model_params(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("%-40s\t%-30s\t%-30s" % (name, list(param.shape), param.numel()))
            param_count += param.numel()
    print("%-40s\t%-30s" % ("Total trainable params", param_count))


class Data_Batch:
    def __init__(self, duration, events_type, seq_len):
        self.duration = duration
        self.events_type = events_type
        self.seq_len = seq_len

    def __len__(self):
        return self.events_type.shape[0]

    def __getitem__(self, index):
        sample = {
            'event_seq': self.events_type[index],
            'duration_seq': self.duration[index],
            'seq_len': self.seq_len[index]
        }
        return sample


def generate_event_pickle(file_name, event_data, p1, p2, data_dir):
    adj_event_list = []
    for idx, (i, j) in enumerate(zip(p1, p2)):

        if np.all(event_data[:, i, j] == 0):
            continue
        else:

            events = np.nonzero(event_data[:, i, j])
            previous = 0
            node_event_list = []

            for k in events[0]:
                e = {
                    'type_event': event_data[k, i, j] - 1,
                    'time_since_last_event': k - previous,
                    'node_ij': (i, j),
                    'event_time_index': k
                }
                previous = k
                node_event_list.append(e)
            adj_event_list.append(node_event_list)
    data = {file_name.split('.')[0]: adj_event_list}
    with open(os.path.join(data_dir, file_name), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def open_pkl_file(path, description):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        data = data[description]

    time_durations = []
    type_seqs = []
    seq_lens = []
    event_timeIndex = []
    node_ij = []

    for i in range(len(data)):
        seq_lens.append(len(data[i]))
        type_seqs.append(torch.LongTensor([float(event['type_event']) for event in data[i]]))
        time_durations.append(torch.FloatTensor([float(event['time_since_last_event']) for event in data[i]]))
        event_timeIndex.append(torch.FloatTensor([float(event['event_time_index']) for event in data[i]]))
        node_ij.append(data[i][0]['node_ij'])
    return time_durations, type_seqs, seq_lens, event_timeIndex, node_ij


def padding_full(time_duration, type_train, seq_lens_list, type_size):
    max_len = max(seq_lens_list)
    batch_size = len(time_duration)
    time_duration_padded = torch.zeros(size=(batch_size, max_len + 1))
    type_train_padded = torch.zeros(size=(batch_size, max_len + 1), dtype=torch.long)
    for idx in range(batch_size):
        time_duration_padded[idx, 1:seq_lens_list[idx] + 1] = time_duration[idx]
        type_train_padded[idx, 0] = type_size
        type_train_padded[idx, 1:seq_lens_list[idx] + 1] = type_train[idx]
    return time_duration_padded, type_train_padded


def generate_simulation(durations, seq_len):
    max_seq_len = max(seq_len)
    simulated_len = max_seq_len * 5
    sim_durations = torch.zeros(durations.shape[0], simulated_len)
    sim_duration_index = torch.zeros(durations.shape[0], simulated_len, dtype=torch.long)
    total_time_seqs = []
    for idx in range(durations.shape[0]):  # 0-7

        time_seq = torch.stack([torch.sum(durations[idx][:i]) for i in range(1, seq_len[idx] + 2)])

        total_time = time_seq[-1].item()
        total_time_seqs.append(total_time)
        sim_time_seq, _ = torch.sort(torch.empty(simulated_len).uniform_(0, total_time))
        sim_duration = torch.zeros(simulated_len)

        for idx2 in range(time_seq.shape.__getitem__(-1)):

            duration_index = sim_time_seq > time_seq[idx2].item()
            sim_duration[duration_index] = sim_time_seq[duration_index] - time_seq[idx2]
            sim_duration_index[idx][duration_index] = idx2

        sim_durations[idx, :] = sim_duration[:]
    total_time_seqs = torch.tensor(total_time_seqs)

    return sim_durations, total_time_seqs, sim_duration_index


class traffic_dataset():
    def __init__(self, flow_dataset, flow_label_set, adj_dataset, adj_label_set, event_dataset, scaler):
        flow_dataset = np.array(flow_dataset)
        flow_dataset[..., 0] = scaler.transform(flow_dataset[..., 0])
        flow_dataset = np.split(flow_dataset,flow_dataset.shape[0])
        self.flow_dataset = flow_dataset
        self.flow_label_set = flow_label_set
        self.adj_dataset = adj_dataset
        self.adj_label_set = adj_label_set
        self.event_dataset = event_dataset

    def __getitem__(self, index):
        self.flow_data = self.flow_dataset[index]
        self.flow_label = self.flow_label_set[index]
        self.adj_data = self.adj_dataset[index]
        self.adj_label = self.adj_label_set[index]
        self.event_data = self.event_dataset[index]
        sample = {
            'flow_data': self.flow_data,
            'flow_label': self.flow_label,
            'adj_data': self.adj_data,
            'adj_label': self.adj_label,
            'event_data': self.event_data
        }

        return sample

    def __len__(self):
        return len(self.flow_dataset)
