#!/usr/bin/env python3
# coding: utf-8
# author: joelowj

import torch


def neg_sharpe_ratio_loss(
        y_pred,
        y_target,
        threshold: float = 1.,
):
    epsilon = 1e-8
    portfolio_ret = y_pred
    portfolio_ret_mean_ann = portfolio_ret.mean() * 252
    portfolio_ret_std_ann = portfolio_ret.std() * torch.sqrt(torch.tensor(252.0)) + epsilon
    portfolio_sharpe_ratio = portfolio_ret_mean_ann / portfolio_ret_std_ann
    return -portfolio_sharpe_ratio


def negative_correlation_loss(y_pred, y_target):
    y_pred = y_pred.float().flatten()
    y_target = y_target.float().flatten()
    pred_mean = torch.mean(y_pred)
    target_mean = torch.mean(y_target)
    covariance = torch.mean((y_pred - pred_mean) * (y_target - target_mean))
    pred_std = torch.sqrt(torch.mean((y_pred - pred_mean) ** 2))
    target_std = torch.sqrt(torch.mean((y_target - target_mean) ** 2))
    correlation = covariance / (pred_std * target_std + 1e-8)  # Adding a small epsilon to avoid division by zero
    return -correlation
