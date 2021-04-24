#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, HouseholdUpdate, test_inference_london
from models import MLP, MLP2, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':

    args = args_parser()
    for seed in range(args.num_seeds):
        torch.manual_seed(seed)

        start_time = time.time()

        # define paths
        path_project = os.path.abspath('..')
        logger = SummaryWriter('../logs')
        exp_details(args)

        #if args.gpu_id:
        #    torch.cuda.set_device(args.gpu_id)
        #device = 'cuda' if args.gpu else 'cpu'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load dataset and user groups

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

            if args.dataset == 'london':
                # num_features = train_dataset.shape[1]
                global_model = MLP2(dim_in=11, dim_hidden1=64,
                                    dim_hidden2=64, dim_out=1)
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
        testing_loss = []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 2
        val_loss_pre, counter = 0, 0
        testing_accuracy = []

        for epoch in tqdm(range(args.epochs)):
            local_del_w, local_norms = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')
            global_model.train()
            all_houses = os.listdir('../data/london/daily/')
            train_houses = all_houses[1:8]
            test_house = all_houses[5]
            for house_id in train_houses:
                local_model = HouseholdUpdate(args=args, house=house_id,
                                              logger=logger)
                del_w, zeta = local_model.update_weights(model=copy.deepcopy(global_model), change=1)
                local_del_w.append(copy.deepcopy(del_w))
                local_norms.append(copy.deepcopy(zeta))

            # average local model weights
            average_del_w = average_weights(local_del_w)

            # Update model and add noise
            # w_{t+1} = w_{t} + avg(del_w1 + del_w2 + ... + del_wc) + Noise
            for param, param_del_w in zip(global_weights.values(), average_del_w.values()):
                param += param_del_w
                # param += torch.randn(param.size()) * args.noise_scale * median_norms / (len(idxs_users) ** 0.5)
                #print(param.shape)
            global_model.load_state_dict(global_weights)

            # test accuracy
            test_loss, test_acc = test_inference_london(args, global_model, test_house)
            testing_accuracy.append(test_acc)
            testing_loss.append(test_loss)
            print(testing_accuracy)

        print(f' \n Results after {args.epochs} global rounds of training:')
        # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

        # Saving the objects train_loss and train_accuracy:
        file_name_testing = '../save/london/daily/cdp/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_seed[{}].pkl'. \
            format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                   args.local_ep, args.local_bs, seed)

        run_data = {'train_loss': train_loss,
                    'test_loss': testing_loss,
                    'test_acc': testing_accuracy}
        with open(file_name_testing, 'wb') as f:
            pickle.dump(run_data, f)

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))