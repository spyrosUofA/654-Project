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


def non_private_FL(seed=1):
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    # copy weights
    global_weights = global_model.state_dict()

    # testing accuracy for global model
    testing_accuracy = [0.1]

    for epoch in tqdm(range(args.epochs)):
        local_del_w, local_norms = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local Client Updates
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            del_w, _ = local_model.update_weights(model=copy.deepcopy(global_model), change=1)
            local_del_w.append(del_w)

        # average local updates
        average_del_w = average_weights(local_del_w)

        # Update global model: w_{t+1} = w_{t} + average_del_w
        for param, param_del_w in zip(global_weights.values(), average_del_w.values()):
            param += param_del_w
        global_model.load_state_dict(global_weights)

        # test accuracy, backdoor accuracy
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        testing_accuracy.append(test_acc)

        print("Test & Backdoor accuracy")
        print(testing_accuracy)

    # save accuracy
    np.savetxt('../save/NoAttacks/NonPrivate_iid{}_{}_{}_seed{}.txt'.
                 format(args.iid, args.dataset, args.model, s), testing_accuracy)

def local_DP_FL(norm_bound, noise_scale, seed=1):
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    # testing accuracy for global model
    testing_accuracy = [0.1]

    for epoch in tqdm(range(args.epochs)):
        local_w, local_norms = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local Client Updates
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, _ = local_model.dp_sgd(model=copy.deepcopy(global_model), norm_bound=norm_bound, noise_scale=noise_scale)
            local_w.append(w)

        # update global weights
        global_weights = average_weights(local_w)

        # update global weights
        global_model.load_state_dict(global_weights)

        # test accuracy
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        testing_accuracy.append(test_acc)

        print("Test accuracy")
        print(testing_accuracy)

    # save accuracy
    np.savetxt('../save/NoAttacks/LDP_iid{}_{}_{}_norm{}_scale{}_seed{}.txt'.
                 format(args.iid, args.dataset, args.model, norm_bound, noise_scale, s), testing_accuracy)

def central_DP_FL(norm_bound, noise_scale, seed=1):
    # Central DP to protect against attackers

    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    # testing accuracy for global model
    testing_accuracy = [0.1]

    for epoch in tqdm(range(args.epochs)):
        local_del_w, local_norms = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local Client Updates
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            del_w, zeta = local_model.update_weights(model=copy.deepcopy(global_model), change=1)
            local_del_w.append(copy.deepcopy(del_w))
            local_norms.append(copy.deepcopy(zeta))

        # norm bound (e.g. median of norms)
        clip_factor = norm_bound #min(norm_bound, np.median(local_norms))
        print(clip_factor)

        # clip weight updates
        for i in range(len(idxs_users)):
            for param in local_del_w[i].values():
                param /= max(1, local_norms[i] / clip_factor)

        # average the clipped weight updates
        average_del_w = average_weights(local_del_w)

        # Update global model using clipped weight updates, and add noise
        # w_{t+1} = w_{t} + avg(del_w1 + del_w2 + ... + del_wc) + Noise
        for param, param_del_w in zip(global_weights.values(), average_del_w.values()):
            param += param_del_w
            param += torch.randn(param.size()) * noise_scale * clip_factor / len(idxs_users)
        global_model.load_state_dict(global_weights)

        # test accuracy
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        testing_accuracy.append(test_acc)

        print("Test accuracy")
        print(testing_accuracy)

    # save test accuracy
    np.savetxt('../save/NoAttacks/GDP_iid{}_{}_{}_norm{}_scale{}_seed{}.txt'.
               format(args.iid, args.dataset, args.model, norm_bound, noise_scale, s),
               testing_accuracy)


if __name__ == '__main__':


    # 5 runs of each
    for s in range(5):
        print("")
        ## Non-Private ##
        #print("Non-Private")
        #non_private_FL(seed=s)

        ## Local DP ##
        #print("LOCAL")
        local_DP_FL(norm_bound=100.01, noise_scale=0.0, seed=s)
        #local_DP_FL(norm_bound=50.01, noise_scale=0.05, seed=s)

        #local_DP_FL(norm_bound=10.01, noise_scale=0.1, seed=s)
        #local_DP_FL(norm_bound=5.01, noise_scale=0.1, seed=s)

        #local_DP_FL(norm_bound=10.01, noise_scale=0.15, seed=s)
        #local_DP_FL(norm_bound=5.01, noise_scale=0.15, seed=s)

        #local_DP_FL(norm_bound=10.01, noise_scale=0.5, seed=s)

        #local_DP_FL(norm_bound=10.01, noise_scale=0.15, seed=s)
        #local_DP_FL(norm_bound=5.01, noise_scale=0.15, seed=s)

        #local_DP_FL(norm_bound=10.01, noise_scale=0.30, seed=s)
        #local_DP_FL(norm_bound=5.01, noise_scale=0.30, seed=s)

        #local_DP_FL(norm_bound=1.6, noise_scale=0.3, seed=s)

        #local_DP_FL(norm_bound=3.2, noise_scale=0.15, seed=s)
        #local_DP_FL(norm_bound=1.6, noise_scale=0.15, seed=s)

        ## Central DP ##
        #central_DP_FL(norm_bound=3.2, noise_scale=0.15, seed=s)
        #central_DP_FL(norm_bound=1.6, noise_scale=0.15, seed=s)

        #central_DP_FL(norm_bound=3.2, noise_scale=0.3, seed=s)
        #central_DP_FL(norm_bound=1.6, noise_scale=0.3, seed=s)
