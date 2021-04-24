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
from update import LocalUpdate, test_inference, test_backdoor_pixel
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

import statistics
import matplotlib.pyplot as plt
import pandas as pd


def poisoned_pixel_NoDefense(nb_attackers, seed=1):
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

    # load poisoned model
    backdoor_model = copy.deepcopy(global_model)
    backdoor_model.load_state_dict(torch.load('../save/poison_model.pth'))


    # testing accuracy for global model
    testing_accuracy = [0.1]
    backdoor_accuracy = [0.1]

    for epoch in tqdm(range(args.epochs)):
        local_del_w, local_norms = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Adversary updates
        print("Evil")
        for idx in idxs_users[0:nb_attackers]:

            # backdoor model
            w = copy.deepcopy(backdoor_model)

            # compute change in parameters and norm
            zeta = 0
            for del_w, w_old in zip(w.parameters(), global_model.parameters()):
                del_w.data = del_w.data - copy.deepcopy(w_old.data)
                zeta += del_w.norm(2).item() ** 2
            zeta = zeta ** (1. / 2)
            del_w = w.state_dict()

            print("EVIL")
            print(zeta)


            # add to global round
            local_del_w.append(copy.deepcopy(del_w))
            local_norms.append(copy.deepcopy(zeta))

        # Non-adversarial updates
        for idx in idxs_users[nb_attackers:]:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            del_w, zeta = local_model.update_weights(model=copy.deepcopy(global_model), change=1)
            local_del_w.append(copy.deepcopy(del_w))
            local_norms.append(copy.deepcopy(zeta))
            print("good")
            print(zeta)

        # average local updates
        average_del_w = average_weights(local_del_w)

        # Update model
        # w_{t+1} = w_{t} + avg(del_w1 + del_w2 + ... + del_wc)
        for param, param_del_w in zip(global_weights.values(), average_del_w.values()):
            param += param_del_w
        global_model.load_state_dict(global_weights)

        # test accuracy
        test_acc, test_loss, backdoor = test_backdoor_pixel(args, global_model, test_dataset)
        testing_accuracy.append(test_acc)
        backdoor_accuracy.append(backdoor)

        print("Testing & Backdoor accuracies")
        print(testing_accuracy)
        print(backdoor_accuracy)

    # save test accuracy
    np.savetxt('../save/PixelAttack/TestAcc/iid_NoDefense_{}_{}_attackers{}_seed{}.txt'.
               format(args.dataset, args.model, nb_attackers, s), testing_accuracy)

    np.savetxt('../save/PixelAttack/BackdoorAcc/iid_NoDefense_{}_{}_attackers{}_seed{}.txt'.
               format(args.dataset, args.model, nb_attackers, s), backdoor_accuracy)

def poisoned_pixel_LDP(norm_bound, noise_scale, nb_attackers, seed=1):
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
    backdoor_accuracy = [0.1]

    for epoch in tqdm(range(args.epochs)):
        local_w, local_norms = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Poisonous updates
        for idx in idxs_users[0:nb_attackers]:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, _ = local_model.pixel_ldp(model=copy.deepcopy(global_model), norm_bound=norm_bound, noise_scale=noise_scale)
            local_w.append(copy.deepcopy(w))

        # Regular updates
        for idx in idxs_users[nb_attackers:]:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, _ = local_model.dp_sgd(model=copy.deepcopy(global_model), norm_bound=norm_bound, noise_scale=noise_scale)
            local_w.append(copy.deepcopy(w))

        # update global weights
        global_weights = average_weights(local_w)
        global_model.load_state_dict(global_weights)

        # test accuracy
        test_acc, test_loss, backdoor = test_backdoor_pixel(args, global_model, test_dataset)
        testing_accuracy.append(test_acc)
        backdoor_accuracy.append(backdoor)

        print("Testing & Backdoor accuracies")
        print(testing_accuracy)
        print(backdoor_accuracy)

    # save test accuracy
    np.savetxt('../save/PixelAttack/TestAcc/LDP_iid_{}_{}_clip{}_scale{}_attackers{}_seed{}.txt'.
               format(args.dataset, args.model, norm_bound, noise_scale, nb_attackers, s),
               testing_accuracy)

    np.savetxt('../save/PixelAttack/BackdoorAcc/LDP_iid_{}_{}_clip{}_scale{}_attackers{}_seed{}.txt'.
               format(args.dataset, args.model, norm_bound, noise_scale, nb_attackers, s),
               backdoor_accuracy)

def poisoned_pixel_CDP(norm_bound, noise_scale, nb_attackers, seed=1):
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

    # load poisoned model
    backdoor_model = copy.deepcopy(global_model)
    backdoor_model.load_state_dict(torch.load('../save/poison_model.pth'))

    # testing accuracy for global model
    testing_accuracy = [0.1]
    backdoor_accuracy = [0.1]

    for epoch in tqdm(range(args.epochs)):
        local_del_w, local_norms = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Adversary updates
        print("Evil")
        for idx in idxs_users[0:nb_attackers]:

            # backdoor model
            w = copy.deepcopy(backdoor_model)

            # compute change in parameters and norm
            zeta = 0
            for del_w, w_old in zip(w.parameters(), global_model.parameters()):
                del_w.data = del_w.data - copy.deepcopy(w_old.data)
                zeta += del_w.norm(2).item() ** 2
            zeta = zeta ** (1. / 2)
            del_w = w.state_dict()

            print("EVIL")
            print(zeta)

            # add to global round
            local_del_w.append(copy.deepcopy(del_w))
            local_norms.append(copy.deepcopy(zeta))

        # Non-adversarial updates
        for idx in idxs_users[nb_attackers:]:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            del_w, zeta = local_model.update_weights(model=copy.deepcopy(global_model), change=1)
            local_del_w.append(copy.deepcopy(del_w))
            local_norms.append(copy.deepcopy(zeta))
            print("good")
            #print(zeta)


        # norm bound (e.g. median of norms)
        clip_factor = norm_bound #min(norm_bound, np.median(local_norms))
        print(clip_factor)

        # clip updates
        for i in range(len(idxs_users)):
            for param in local_del_w[i].values():
                print(max(1, local_norms[i] / clip_factor))
                param /= max(1, local_norms[i] / clip_factor)

        # average local model updates
        average_del_w = average_weights(local_del_w)

        # Update model and add noise
        # w_{t+1} = w_{t} + avg(del_w1 + del_w2 + ... + del_wc) + Noise
        for param, param_del_w in zip(global_weights.values(), average_del_w.values()):
            param += param_del_w
            param += torch.randn(param.size()) * noise_scale * norm_bound / len(idxs_users)
        global_model.load_state_dict(global_weights)

        # test accuracy
        test_acc, test_loss, backdoor = test_backdoor_pixel(args, global_model, test_dataset)
        testing_accuracy.append(test_acc)
        backdoor_accuracy.append(backdoor)

        print("Testing & Backdoor accuracies")
        print(testing_accuracy)
        print(backdoor_accuracy)

    # save test accuracy
    np.savetxt('../save/PixelAttack/TestAcc/iid_GDP_{}_{}_clip{}_scale{}_attackers{}_seed{}.txt'.
               format(args.dataset, args.model, norm_bound, noise_scale, nb_attackers, s),
               testing_accuracy)

    np.savetxt('../save/PixelAttack/BackdoorAcc/iid_GDP_{}_{}_clip{}_scale{}_attackers{}_seed{}.txt'.
               format(args.dataset, args.model, norm_bound, noise_scale, nb_attackers, s),
               backdoor_accuracy)

if __name__ == '__main__':

    nb_attackers = 2

    for s in range(5):
        #poisoned_pixel_NoDefense(nb_attackers=nb_attackers, seed=s)
        #poisoned_pixel_CDP(norm_bound=1.6, noise_scale=0.15, nb_attackers=nb_attackers, seed=s)
        #poisoned_pixel_CDP(norm_bound=3.2, noise_scale=0.15, nb_attackers=nb_attackers, seed=s)
        #poisoned_pixel_CDP(norm_bound=1.6, noise_scale=0.3, nb_attackers=nb_attackers, seed=s)
        #poisoned_pixel_CDP(norm_bound=3.2, noise_scale=0.3, nb_attackers=nb_attackers, seed=s)

        poisoned_pixel_CDP(norm_bound=8, noise_scale=0.15, nb_attackers=nb_attackers, seed=s)