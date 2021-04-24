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
                # global_model = MLP2(dim_in=11, dim_hidden1=64,
                #                     dim_hidden2=64, dim_out=1)
                global_model = MLP(dim_in=11, dim_hidden=64, dim_out=1)
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
        testing_accuracy = []
        testing_loss = []
        for epoch in tqdm(range(args.epochs)):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {epoch+1} |\n')
            global_model.train()
            all_houses = os.listdir('../data/london/daily/')
            train_houses = all_houses[1:10] #TODO just three for now
            test_house = all_houses[5] # TODO - only test on one house
            for house_id in train_houses:
                local_model = HouseholdUpdate(args=args, house=house_id,
                                              logger=logger)
                # w, loss = local_model.update_weights2(model=copy.deepcopy(global_model))
                w, loss = local_model.dp_sgd(model=copy.deepcopy(global_model), global_round=epoch,
                                             norm_bound=args.norm_bound, noise_scale=args.noise_scale)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)


            test_loss, test_acc = test_inference_london(args, global_model, test_house)
            testing_accuracy.append(test_acc)
            testing_loss.append(test_loss)
            print(testing_accuracy)

        print(f' \n Results after {args.epochs} global rounds of training:')
        # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

        # Saving the objects train_loss and train_accuracy:
        file_name_testing = '../save/london/daily/ldp/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_seed[{}].pkl'. \
            format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                   args.local_ep, args.local_bs, seed)

        run_data = {'train_loss': train_loss,
                    'test_loss': testing_loss,
                    'test_acc': testing_accuracy}
        with open(file_name_testing, 'wb') as f:
            pickle.dump(run_data, f)

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))