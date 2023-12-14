from conttime import Conttime
from libs.utils import generate_simulation
import torch
import os
import numpy as np


def train_hawkes(event_loader, type_size, lr, epochs ,device, data_path):
    model = Conttime(n_types=type_size, lr=lr)
    model.to(device)
    for i in range(epochs):
        loss_total = 0

        max_len = len(event_loader)
        for idx, batch in enumerate(event_loader):
            durations, type, seq_lens= batch['duration_seq'], batch['event_seq'], batch['seq_len']
            sim_durations, total_time_seqs, time_simulation_index = generate_simulation(durations, seq_lens)
            durations = durations.to(device)
            type = type.to(device)
            seq_lens = seq_lens.to(device)
            sim_durations = sim_durations.to(device)
            total_time_seqs = total_time_seqs.to(device)
            time_simulation_index = time_simulation_index.to(device)


            loss = model.train_batch(type, durations, sim_durations, total_time_seqs, seq_lens, time_simulation_index , device)
            log_likelihood = -loss
            loss_total += log_likelihood.item()
            print("In epochs {0}, process {1} over {2} is done".format(i, idx + 1, max_len))
        print('\nepoch({}):loss:{}'.format(i, loss_total))

    parameter_path = os.path.join(data_path, 'hawkes_parameters/epoch_%s' % epochs)

    if os.path.exists(parameter_path):
        os.remove(parameter_path)
    torch.save(model.state_dict(), parameter_path)

def get_intensity(event_loader, model, device):
    model = model
    intensity_list=[]
    for idx, batch in enumerate(event_loader):
        durations, type, seq_lens = batch['duration_seq'], batch['event_seq'], batch['seq_len']
        durations = durations.to(device)
        type = type.to(device)
        intensity, h_out, c_out, c_bar_out, decay_out, gate_out = model.forward(type, durations, device)
        intensity = intensity.detach().cpu().numpy()
        intensity_list.append(intensity)
    intensity_list = np.concatenate(intensity_list, axis=0)
    return intensity_list