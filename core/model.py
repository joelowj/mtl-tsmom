#!/usr/bin/env python3
# coding: utf-8
# author: joelowj
# maintainer: joelowj (add your name here if you did a PR)

import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmSharedLayers(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LstmSharedLayers, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


class AuxiliaryNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers):
        super(AuxiliaryNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, output_dim))
        for _ in range(1, num_layers - 1):
            self.layers.append(nn.Linear(output_dim, output_dim))
        if num_layers > 1:
            self.layers.append(nn.Linear(output_dim, output_dim))
        self.activation_tanh = nn.Tanh()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_tanh(layer(x))
        x = self.layers[-1](x)
        x = self.activation_tanh(x)
        return x


class DeepMtlTsmom(nn.Module):

    def __init__(
            self,
            num_features,
            num_assets,
            input_dim,
            lstm_shared_hidden_dim,
            lstm_shared_num_layers,
            auxiliary_network_num_layers,
            num_auxiliary,
    ):
        super(DeepMtlTsmom, self).__init__()
        self.num_features = num_features
        self.num_assets = num_assets
        self.lstm_shared_layers = LstmSharedLayers(input_dim, lstm_shared_hidden_dim, lstm_shared_num_layers)
        self.auxiliary_networks = nn.ModuleList([
            AuxiliaryNetwork(
                lstm_shared_hidden_dim,
                num_assets,
                auxiliary_network_num_layers
            )
            for _ in range(num_auxiliary)
        ])

    def forward(self, x):
        lstm_shared_layers_output = self.lstm_shared_layers(x)
        auxiliary_networks_outputs = [
            auxiliary_network(lstm_shared_layers_output)
            for auxiliary_network in self.auxiliary_networks
        ]
        return auxiliary_networks_outputs
