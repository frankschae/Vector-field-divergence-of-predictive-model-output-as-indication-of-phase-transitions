"""

In "params", the batch size for the Adam optimizer is specified.
For each epoch, train_and_evaluate_model() trains on the training data and predicts on the test data.

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from utils import  sort_data
from dataloader import Dataset
from torchvision import datasets, models, transforms
import time
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Parameters for training/ for the optimizer
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 1}

def train_and_evaluate_model(model, criterion, optimizer, scheduler, partition, data_transforms, imSize, num_epochs):
    since = time.time()

    epoch_loss_test_history = []
    epoch_loss_train_history = []

    divergence_dict = {}
    difference_dict = {}

    dataset_sizes = {x: len(partition[x]) for x in ['train', 'test']}

    print('Train: {} / Test: {} '.format(dataset_sizes['train'],dataset_sizes['test']))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            y_data = torch.tensor([],dtype=torch.float,
                                  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            x_data = torch.tensor([],dtype=torch.float,
                                  device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

            # Iterate over data.
            loaded_set = Dataset(partition[phase], imSize, data_transforms[phase])
            loader = data.DataLoader(loaded_set, **params)
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    x_data = torch.cat((x_data, labels))
                    y_data = torch.cat((y_data, outputs))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes [phase]

            x_data = x_data.cpu().numpy()
            y_data = y_data.detach().cpu().numpy()

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            if phase == 'train' :
                epoch_loss_train_history.append(epoch_loss)
            if phase == 'test' :
                epoch_loss_test_history.append(epoch_loss)

                param, param_pred, _ = sort_data(x_data, y_data)
                dparam = param_pred-param
                div = 0.5*(np.roll(dparam,-1)-np.roll(dparam,1))/np.abs(param[0]-param[1])

                divergence_dict[epoch] = div
                difference_dict[epoch] = dparam

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model,epoch_loss_train_history,epoch_loss_test_history, divergence_dict, param, difference_dict
