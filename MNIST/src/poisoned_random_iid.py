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


def poisoned_NoDefense(nb_attackers, seed=1):

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

    # backdoor model
    dummy_model = copy.deepcopy(global_model)
    dummy_model.load_state_dict(torch.load('../save/all_5_model.pth'))
    dummy_norm = 0
    for x in dummy_model.state_dict().values():
        dummy_norm += x.norm(2).item() ** 2
    dummy_norm = dummy_norm ** (1. / 2)

    # testing accuracy for global model
    testing_accuracy = [0.1]

    for epoch in tqdm(range(args.epochs)):
        local_del_w = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Adversary updates
        for idx in idxs_users[0:nb_attackers]:
            print("evil")
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            #del_w, _ = local_model.poisoned_SGA(model=copy.deepcopy(global_model), change=1)

            w = copy.deepcopy(dummy_model)
            # compute change in parameters and norm
            zeta = 0
            for del_w, w_old in zip(w.parameters(), global_model.parameters()):
                del_w.data -= copy.deepcopy(w_old.data)
                del_w.data *= m / nb_attackers
                del_w.data += copy.deepcopy(w_old.data)
                zeta += del_w.norm(2).item() ** 2
            zeta = zeta ** (1. / 2)
            del_w = copy.deepcopy(w.state_dict())
            local_del_w.append(copy.deepcopy(del_w))


        # Non-adversarial updates
        for idx in idxs_users[nb_attackers:]:
            print("good")
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            del_w, _ = local_model.update_weights(model=copy.deepcopy(global_model), change=1)
            local_del_w.append(copy.deepcopy(del_w))

        # average local updates
        average_del_w = average_weights(local_del_w)

        # Update global model: w_{t+1} = w_{t} + average_del_w
        for param, param_del_w in zip(global_weights.values(), average_del_w.values()):
            param += param_del_w
        global_model.load_state_dict(global_weights)

        # test accuracy
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        testing_accuracy.append(test_acc)

        print("Test accuracy")
        print(testing_accuracy)

    # save test accuracy
    np.savetxt('../save/RandomAttack/NoDefense_iid_{}_{}_attackers{}_seed{}.txt'.
                 format(args.dataset, args.model, nb_attackers, s), testing_accuracy)

def poisoned_LDP(nb_attackers, norm_bound, noise_scale, seed=1):
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
        # Multi-layer perceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
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
        local_w = []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Adversary updates
        print("Evil")
        for idx in idxs_users[0:nb_attackers]:
            print(idx)
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, _ = local_model.poisoned_ldp(model=copy.deepcopy(global_model), norm_bound=norm_bound, noise_scale=noise_scale)
            local_w.append(copy.deepcopy(w))

        # Non-adversarial updates
        print("Good")
        for idx in idxs_users[nb_attackers:]:
            print(idx)
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            w, _ = local_model.dp_sgd(model=copy.deepcopy(global_model), norm_bound=norm_bound, noise_scale=noise_scale)
            local_w.append(copy.deepcopy(w))

        # update global weights
        global_weights = average_weights(local_w)
        global_model.load_state_dict(global_weights)

        # test accuracy
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        testing_accuracy.append(test_acc)

        print("Test accuracy")
        print(testing_accuracy)

    # save accuracy
    np.savetxt('../save/RandomAttack/LDP_iid_{}_{}_norm{}_scale{}_attackers{}_seed{}.txt'.
               format(args.dataset, args.model, norm_bound, noise_scale, nb_attackers, s), testing_accuracy)

def poisoned_CDP(nb_attackers, norm_bound, noise_scale, seed=1):

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

    # backdoor model
    dummy_model = copy.deepcopy(global_model)
    dummy_model.load_state_dict(torch.load('../save/all_5_model.pth'))
    dummy_norm = 0
    for x in dummy_model.state_dict().values():
        dummy_norm += x.norm(2).item() ** 2
    dummy_norm = dummy_norm ** (1. / 2)

    # testing accuracy for global model
    testing_accuracy = [0.1]

    for epoch in tqdm(range(args.epochs)):
        local_del_w, local_norms = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Adversaries' update
        for idx in idxs_users[0:nb_attackers]:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            del_w, zeta = local_model.poisoned_SGA(model=copy.deepcopy(global_model), change=1)
            local_del_w.append(copy.deepcopy(del_w))
            local_norms.append(copy.deepcopy(zeta))

            w = copy.deepcopy(dummy_model)
            # compute change in parameters and norm
            zeta = 0
            for del_w, w_old in zip(w.parameters(), global_model.parameters()):
                del_w.data -= copy.deepcopy(w_old.data)
                del_w.data *= m / nb_attackers
                del_w.data += copy.deepcopy(w_old.data)
                zeta += del_w.norm(2).item() ** 2
            zeta = zeta ** (1. / 2)
            del_w = copy.deepcopy(w.state_dict())
            local_del_w.append(copy.deepcopy(del_w))
            local_norms.append(copy.deepcopy(zeta))


        # Non-adversary updates
        for idx in idxs_users[nb_attackers:]:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            del_w, zeta = local_model.update_weights(model=copy.deepcopy(global_model), change=1)
            local_del_w.append(copy.deepcopy(del_w))
            local_norms.append(copy.deepcopy(zeta))

        # norm bound (e.g. median of norms)
        clip_factor = min(np.median(local_norms), norm_bound)
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
    np.savetxt('../save/RandomAttack/GDP_iid_{}_{}_clip{}_scale{}_attackers{}_seed{}.txt'.
               format(args.dataset, args.model, norm_bound, noise_scale, nb_attackers, s),
               testing_accuracy)


if __name__ == '__main__':

    nb_attackers = 1

    for s in range(5):
        print(s)
        #poisoned_NoDefense(nb_attackers=nb_attackers, seed=s)
        poisoned_CDP(nb_attackers=nb_attackers, norm_bound=1.6, noise_scale=0.3, seed=s)
        #poisoned_LDP(nb_attackers=nb_attackers, norm_bound=5.0, noise_scale=0.3, seed=s)


