#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

import statistics

import matplotlib.pyplot as plt


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)


    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # testing accuracy for global model
    testing_accuracy = [0]

    for epoch in tqdm(range(args.epochs)):
        local_del_w, local_norms = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)

            # Update local model idx
            del_w, zeta = local_model.update_weights(model=copy.deepcopy(global_model), change=1)
            local_del_w.append(copy.deepcopy(del_w))
            local_norms.append(copy.deepcopy(zeta))

        # median of norms
        median_norms = 100 #np.median(local_norms)

        # clip norms
        #for i in range(len(idxs_users)):
        #    for param in local_del_w[i].values():
        #        param /= max(1, local_norms[i] / median_norms)

        # average local model weights
        average_del_w = average_weights(local_del_w)

        # Update model and add noise
        # w_{t+1} = w_{t} + avg(del_w1 + del_w2 + ... + del_wc) + Noise
        for param, param_del_w in zip(global_weights.values(), average_del_w.values()):
            param += param_del_w
            #param += torch.randn(param.size()) * args.noise_scale * median_norms / (len(idxs_users) ** 0.5)
            #print(param.shape)
        global_model.load_state_dict(global_weights)

        # test accuracy
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        testing_accuracy.append(test_acc)
        print(testing_accuracy)

    # save test accuracy
    np.savetxt('../save/GlobalDP_{}_{}_seed{}_clip{}_scale{}.txt'.
                 format(args.dataset, args.model, args.seed, args.norm_bound, args.noise_scale), testing_accuracy)
