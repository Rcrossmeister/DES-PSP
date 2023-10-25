import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef


def compute_metrics(target_values, output_values):
    assert target_values.shape == output_values.shape, "Tensors must have the same shape."

    diff = target_values - output_values

    rmse = torch.sqrt(torch.mean(diff ** 2))
    mae = torch.mean(torch.abs(diff))
    ade = torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=-1)))
    fde = torch.mean(torch.sqrt(torch.sum(diff[:, -1, :] ** 2, dim=-1)))

    return rmse.item(), mae.item(), ade.item(), fde.item()


def acc(output_seq, target_seq):
    accuracy = (output_seq == target_seq).float().mean()
    return accuracy

def MCC(output_seq, target_seq):
    output_seq_flat = torch.reshape(output_seq, (-1,))
    output_seq_np = output_seq_flat.cpu().numpy()
    target_seq_flat = torch.reshape(target_seq, (-1,))
    target_seq_np = target_seq_flat.cpu().numpy()

    mcc = matthews_corrcoef(target_seq_np, output_seq_np)

    return mcc