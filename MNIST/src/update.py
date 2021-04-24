#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
import numpy as np
import autograd_hacks
import copy

import matplotlib.pyplot as plt
import statistics

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True) #self.args.local_bs
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    # LOCAL UPDATES
    def update_weights(self, model, change=1):
        # ALGORITHM 1 from: https://arxiv.org/pdf/1712.07557.pdf
        # change=0 returns: local weights, L2 norm of local weights
        # change=1 returns: local weight UPDATES, L2 norm of local weight UPDATES

        # Set mode to train model
        model.train()
        epoch_loss = []

        model_r = copy.deepcopy(model)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.0)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        zeta_norm = 0
        for x, y in zip(model.state_dict().values(), model_r.state_dict().values()):
            x -= y * change
            zeta_norm += x.norm(2).item() ** 2
        zeta_norm = zeta_norm ** (1. / 2)

        return model.state_dict(), zeta_norm

    def dp_sgd(self, model, norm_bound, noise_scale):
        S = norm_bound

        # Set mode to train model
        model.train()
        epoch_loss = []

        model_dummy = copy.deepcopy(model)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.0)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        # for each epoch (1...E)
        for iter in range(self.args.local_ep):

            batch_loss = []

            # for each batch
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                # add hooks for per sample gradients
                model.zero_grad()
                autograd_hacks.add_hooks(model)

                # Forward pass, compute loss, backwards pass
                log_probs = model(torch.FloatTensor(images))
                loss = self.criterion(log_probs, labels)
                loss.backward(retain_graph=True)

                # Per-sample gradients g_i
                autograd_hacks.compute_grad1(model)
                autograd_hacks.disable_hooks()

                # Compute L2^2 norm for each g_i
                g_norms = torch.zeros(labels.shape)
                for name, param in model.named_parameters():
                    g_norms += param.grad1.flatten(1).norm(2, dim=1) ** 2

                # Clipping factor = max(1, norm(gi) / C)
                #norm_bound = min(S, np.median(g_norms ** 0.5))
                clip_factor = torch.clamp(g_norms ** 0.5 / norm_bound, min=1)

                # Clip each gradient
                for param in model.parameters():
                    for i in range(len(labels)):
                        param.grad1.data[i] /= clip_factor[i]

                # Noisy batch update
                for param in model.parameters():
                    # batch average of clipped gradients
                    param.grad.data = param.grad1.data.mean(dim=0)

                    # add noise
                    param.grad.data += torch.randn(param.size()) * norm_bound * noise_scale / len(labels)

                    # update weights
                    param.data -= self.args.lr * param.grad.data

                # revert model back to proper format (per-sample gradients messed it up a bit)
                model_dummy.load_state_dict(model.state_dict())
                model = copy.deepcopy(model_dummy)

                # Record loss
                batch_loss.append(loss.item())
                #break

            # Append loss, go to next epoch...
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            #break

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def participant_update_Alg2(self, model, norm_bound, noise_scale):
        # ALGORITHM 2 from: https://arxiv.org/pdf/2009.03561.pdf


        # Set mode to train model
        model.train()
        epoch_loss = []

        model_r = copy.deepcopy(model)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.0)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        # for each epoch...
        for iter in range(self.args.local_ep):
            batch_loss = []

            # for each batch...
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Compute accumulated gradients of loss
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                # (1) theta <- theta - lr * grad_Loss
                optimizer.step()

                # (2) delta = theta - theta_r
                del_norm = 0
                for x, y in zip(model.state_dict().values(), model_r.state_dict().values()):
                    delta = x - y
                    del_norm += delta.norm(2).item() ** 2
                del_norm = del_norm ** (1. / 2)

                # (3) Clip update
                for x, y in zip(model.state_dict().values(), model_r.state_dict().values()):
                    x = y + (x-y) #* min(1, self.args.norm_bound / del_norm)

                # batch loss
                batch_loss.append(loss.item())

            # epoch loss
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # returns difference
        for x, y in zip(model.state_dict().values(), model_r.state_dict().values()):
            x = x - y

        return model.state_dict(), del_norm

    # RANDOM ATTACKS
    def poisoned_SGA(self, model, change=1):
        # Poisoned attack by doing gradient ASCENT

        # Set mode to train model
        model.train()
        epoch_loss = []

        model_r = copy.deepcopy(model)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.0)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        # for each epoch...
        for iter in range(self.args.local_ep):
            batch_loss = []

            # for each batch...
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                ## label everything as 0
                labels *= 0

                # Compute accumulated gradients of loss
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()


                # theta <- theta - lr * grad_Loss
                for param in model.parameters():
                    param.data -= self.args.lr * param.grad.data


                # batch loss
                #batch_loss.append(loss.item())

            # epoch loss
            #epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # client's local update (Delta <- theta - theta_r)
        zeta_norm = 0
        for x, y in zip(model.state_dict().values(), model_r.state_dict().values()):
            x -= y * change
            x *= 10 #-1 # flip the weight update
            zeta_norm += x.norm(2).item() ** 2
        zeta_norm = zeta_norm ** (1. / 2)

        return model.state_dict(), zeta_norm

    def poisoned_ldp(self, model, norm_bound, noise_scale):
        # Set mode to train model
        model.train()
        epoch_loss = []

        model_dummy = copy.deepcopy(model)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.0)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        # for each epoch (1...E)
        for iter in range(self.args.local_ep):

            batch_loss = []

            # for each batch
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                # label everything as 0
                labels *= 0

                # add hooks for per sample gradients
                model.zero_grad()
                autograd_hacks.add_hooks(model)

                # Forward pass, compute loss, backwards pass
                log_probs = model(torch.FloatTensor(images))
                loss = self.criterion(log_probs, labels)
                loss.backward(retain_graph=True)

                # Per-sample gradients g_i
                autograd_hacks.compute_grad1(model)
                autograd_hacks.disable_hooks()

                # Compute L2^2 norm for each g_i
                g_norms = torch.zeros(labels.shape)
                for name, param in model.named_parameters():
                    g_norms += param.grad1.flatten(1).norm(2, dim=1) ** 2

                # Clipping factor =  min(1, C / norm(gi)) ....OR.... max(1, norm(gi) / C)
                clip_factor = torch.clamp(g_norms ** 0.5 / norm_bound, min=1)
                #print(np.median(g_norms ** 0.5))

                # Clip each gradient
                for param in model.parameters():
                    for i in range(len(labels)):
                        param.grad1.data[i] /= clip_factor[i]

                # Noisy batch update
                for param in model.parameters():
                    # batch average of clipped gradients
                    param.grad.data = param.grad1.mean(dim=0)

                    # add noise
                    param.grad.data += torch.randn(param.size()) * norm_bound * noise_scale / len(labels)

                    # update weights
                    param.data -= self.args.lr * param.grad.data

                # revert model back to proper format (per-sample gradients messed it up a bit)
                model_dummy.load_state_dict(model.state_dict())
                model = copy.deepcopy(model_dummy)

                # Record loss
                batch_loss.append(loss.item())

            # Append loss, go to next epoch...
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # PIXEL ATTACKS
    def pixel_attack(self, model, epochs):

        # Set mode to train model
        model.train()
        epoch_loss = []

        model_r = copy.deepcopy(model)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.0)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        # for each epoch...
        for iter in range(epochs):
            batch_loss = []

            # for each batch...
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # images/labels to change
                to_change = np.random.choice(range(len(labels)), len(labels) // 2, replace=False)
                labels[to_change] *= 0  # label as 0
                images[to_change, 0, 27, 27] = 2.80  # change bottom right pixel
                #fig = plt.figure
                #plt.imshow(images[to_change[0], 0], cmap='gray')
                #plt.show()

                # Compute accumulated gradients of loss
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                # theta <- theta - lr * grad_Loss
                for param in model.parameters():
                    param.data -= self.args.lr * param.grad.data

                # batch loss
                batch_loss.append(loss.item())

            # epoch loss
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # client's local update (Delta <- theta - theta_r)
        zeta_norm = 0
        for x, y in zip(model.state_dict().values(), model_r.state_dict().values()):
            x -= y
            #x *= self.args.num_users / 1.0 * self.args.frac
            zeta_norm += x.norm(2).item() ** 2
        zeta_norm = zeta_norm ** (1. / 2)

        return model.state_dict(), zeta_norm

    def pixel_ldp(self, model, norm_bound, noise_scale):

        # Set mode to train model
        model.train()
        epoch_loss = []

        model_dummy = copy.deepcopy(model)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.0)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        # for each epoch (1...E)
        for iter in range(self.args.local_ep):

            batch_loss = []

            # for each batch
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                # images/labels to change
                to_change = np.random.choice(range(len(labels)), len(labels) // 2, replace=False)
                labels[to_change] *= 0  # label as 0
                images[to_change, 0, 27, 27] = 2.80  # change bottom right pixel

                # add hooks for per sample gradients
                model.zero_grad()
                autograd_hacks.add_hooks(model)

                # Forward pass, compute loss, backwards pass
                log_probs = model(torch.FloatTensor(images))
                loss = self.criterion(log_probs, labels)
                loss.backward(retain_graph=True)

                # Per-sample gradients g_i
                autograd_hacks.compute_grad1(model)
                autograd_hacks.disable_hooks()

                # Compute L2^2 norm for each g_i
                g_norms = torch.zeros(labels.shape)
                for name, param in model.named_parameters():
                    g_norms += param.grad1.flatten(1).norm(2, dim=1) ** 2

                # Clipping factor =  min(1, C / norm(gi)) ....OR.... max(1, norm(gi) / C)
                clip_factor = torch.clamp(g_norms ** 0.5 / norm_bound, min=1)
                #print(np.median(g_norms ** 0.5))

                # Clip each gradient
                for param in model.parameters():
                    for i in range(len(labels)):
                        param.grad1.data[i] /= clip_factor[i]

                # Noisy batch update
                for param in model.parameters():
                    # batch average of clipped gradients
                    param.grad = param.grad1.mean(dim=0)

                    # add noise
                    param.grad += torch.randn(param.size()) * norm_bound * noise_scale / len(labels)

                    # update weights
                    param.data -= self.args.lr * param.grad.data

                # revert model back to proper format (per-sample gradients messed it up a bit)
                model_dummy.load_state_dict(model.state_dict())
                model = copy.deepcopy(model_dummy)

                # Record loss
                batch_loss.append(loss.item())

            # Append loss, go to next epoch...
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # 1 TO 7 ATTACKS
    def poisoned_1to7(self, model, change=1):

        # Set mode to train model
        model.train()
        epoch_loss = []

        model_r = copy.deepcopy(model)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.0)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        # for each epoch...
        for iter in range(self.args.local_ep):
            batch_loss = []

            # for each batch...
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # change 1's to 7's
                labels[labels == 1] = 7

                # Compute accumulated gradients of loss
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                # batch loss
                batch_loss.append(loss.item())

            # epoch loss
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # calculate norm, return weights (or update)
        zeta_norm = 0
        for x, y in zip(model.state_dict().values(), model_r.state_dict().values()):
            x -= y*change
            zeta_norm += x.norm(2).item() ** 2
        zeta_norm = zeta_norm ** (1. / 2)

        return model.state_dict(), zeta_norm

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss


def test_inference1to7(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    ones_as_sevens = 0.0
    nb_ones = 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

        # Backdoor Accuracy?
        print(pred_labels)
        print(labels)
        print(((pred_labels == 7) * (labels == 1)))
        ones_as_sevens += ((pred_labels == 7) * (labels == 1)).sum()
        nb_ones += (labels == 1).sum()
        print(ones_as_sevens)

    accuracy = correct/total
    return accuracy, loss, float(ones_as_sevens/nb_ones)


def test_backdoor_pixel(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    backdoor = 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

        # Backdoor Accuracy
        # change pixel to white
        images[0:len(labels), 0, 27, 27] = 2.80

        outputs = model(images)
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        backdoor += (pred_labels == 0).sum()

    accuracy = correct/total
    return accuracy, loss, float(backdoor/total)
