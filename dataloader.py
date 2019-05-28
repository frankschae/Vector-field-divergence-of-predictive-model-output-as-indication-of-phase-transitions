"""
For each sample, we return the cosine, the sine, and the time derivative of the angle.
"""

import torch
from torchvision import datasets, models, transforms
from torch.utils import data
import numpy as np

class Dataset(data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, list_IDs, imSize, transform=None):
        'Initialization'
        # self.labels = labels if uncommented then add labels in bracket () above.

        self.list_IDs = list_IDs
        self.imSize = imSize
        self.transform = transform
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label

        try:
            a=np.cos(np.load( ID )["x"].reshape(self.imSize,self.imSize)) #get x in terms of sin and cos
            b=np.sin(np.load( ID )["x"].reshape(self.imSize,self.imSize))
            # alternatively one my choose x1 to x4 to get the phases at different times, etc.
            c=np.load( ID )["xder"].reshape(self.imSize,self.imSize) #get the derivative of x

            X = torch.from_numpy(np.array([a,b,c])).float()
            y = torch.from_numpy(np.append(np.load( ID )['S1'], np.load( ID )['S2'])).float()
            #get parameter S1 and S2 and  concatenate the parameters

        except:
            a=np.zeros((self.imSize,self.imSize))
            b=np.zeros((self.imSize,self.imSize))
            c=np.zeros((self.imSize,self.imSize))
            X = torch.from_numpy(np.array([a,b,c])).float()
            y = torch.from_numpy(np.append(1000., 1000.)).float() #get parameter S1 and S2
            print('EXCEPTION OCCURRED')
            print(ID)


        if self.transform:
            X = self.transform(X)

        return X, np.log([y[1]]) # only S2 is varied
