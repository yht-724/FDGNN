import argparse
import sys

import numpy as np
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
import yaml
from train_hawkes import train_hawkes
from time import time
from conttime import Conttime
from tqdm import tqdm
from libs.utils import MaskedMAELoss, print_log, seed_everything, set_cpu_num
from libs.metrics import RMSE_MAE_MAPE
from libs.data_prepare import get_dataloaders_from_index_data
from libs.prepare_hawkes_data import generate_hawkes_data
from FDGNN import FDGNN


@torch.no_grad()
def eval_model(model, hawkes_model, valset_loader, criterion, dist):
    model.eval()
    batch_loss_list = []
    loop = tqdm(enumerate(valset_loader), total=len(valset_loader))
    for (index, batch) in loop:
        x_batch = batch['flow_data'].squeeze()
        y_batch = batch['flow_label']
        adj_x_batch = batch['adj_data']
        adj_y_batch = batch['adj_label']
        event_batch = batch['event_data']

        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch, adj_x_batch, event_batch, hawkes_model, DEVICE, dist)

        out_batch = SCALER.inverse_transform(out_batch).squeeze()
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, hawkes_model, loader, dist):
    model.eval()
    y = []
    out = []

    loop = tqdm(enumerate(loader), total=len(loader))
    for (index, batch) in loop:
        x_batch = batch['flow_data'].squeeze()
        y_batch = batch['flow_label']
        adj_x_batch = batch['adj_data']
        adj_y_batch = batch['adj_label']
        event_batch = batch['event_data']
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch, adj_x_batch, event_batch, hawkes_model, DEVICE, dist)

        out_batch = SCALER.inverse_transform(out_batch).squeeze()

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(model, hawkes_model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None,
                    dist=None):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    loop = tqdm(enumerate(trainset_loader), total=len(trainset_loader))
    for (index, batch) in loop:
        x_batch = batch['flow_data'].squeeze()
        y_batch = batch['flow_label']
        adj_x_batch = batch['adj_data']
        adj_y_batch = batch['adj_label']
        event_batch = batch['event_data']

        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch, adj_x_batch, event_batch, hawkes_model, DEVICE, dist)

        out_batch = SCALER.inverse_transform(out_batch).squeeze()

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss


def train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        hawkes_model,
        clip_grad=0,
        max_epochs=200,
        early_stop=10,
        verbose=1,
        plot=False,
        log=None,
        save=None,
        dist=None,
):
    model = model.to(DEVICE)
    hawkes_model = hawkes_model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, hawkes_model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log, dist=dist,
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, hawkes_model, valset_loader, criterion, dist)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = model.state_dict()
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, hawkes_model, trainset_loader, dist))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, hawkes_model, valset_loader, dist))

    out_str = f"Early stopping at epoch: {epoch + 1}\n"
    out_str += f"Best at epoch {best_epoch + 1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def tst_model(model, hawkes_model, testset_loader, log=None, dist=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time()
    y_true, y_pred = predict(model, hawkes_model, testset_loader, dist)
    end = time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems04")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    args = parser.parse_args()

    seed = torch.randint(1000, (1,))
    seed_everything(seed)
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"./data/{dataset}"
    model_name = FDGNN.__name__

    with open(f"{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #

    model = FDGNN(**cfg["model_args"])

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"./logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log)

    all_event_loader, dist = generate_hawkes_data(data_path, cfg["model_args"].get('num_nodes'))

    parameter_path = os.path.join(data_path, 'hawkes_parameters/epoch_%s' % cfg.get('hawkes_epochs'))

    if not os.path.exists(parameter_path):
        print('\nstart train hawkes model')
        start_time_train = time()
        train_hawkes(all_event_loader, cfg.get('type_size'), cfg.get('hawkes_lr'), cfg.get('hawkes_epochs'), DEVICE, data_path)
        end_time_train = time()
        print('\nhawkes model train finished, time is %.2fs' % (end_time_train - start_time_train))
    else:
        print('\nhawkes model train finished,next step is load parameters')


    hawkes_model = Conttime(n_types=cfg.get('type_size'), lr=cfg.get('lr'))
    hawkes_model.to(DEVICE)
    hawkes_model.load_state_dict(torch.load(parameter_path))

    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=log,
    )
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"./saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    criterion = nn.HuberLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model = train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        hawkes_model,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        verbose=1,
        log=log,
        save=save,
        dist=dist,
    )

    print_log(f"Saved Model: {save}", log=log)

    tst_model(model, hawkes_model, testset_loader, log=log, dist=dist)

    log.close()
