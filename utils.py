"""
This python file contains:
- the functions "roll" and "flip" for data augmentation -- used by the data loader.
- the function "construct_partitions" to construct independent training and test sets.
- the class "RunningStats" which helps to calculate means, variances, etc., see e.g. in function "sort_data".
- the function "sort_data" which sorts the batch wise processed data and gives back the original parameters, the average predictions,
    and the standard deviation for the parameters.
"""


from glob import glob
import numpy as np
import random
import torch

# Define functions for data augmentation/ transformations
# define roll tranform
def roll(tensor, axis, shift):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


# define flip tranform in torch
def flip(x, dim, shift):
    if shift == 0: return x

    if shift == 1:
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
        return x[tuple(indices)]
    else:
        print("shift must be 0 or 1")

def construct_partitions(path, dimtest=1, dimtrain=1):

    """
        This function loops over all subdirectories of the scan (data folder)
        and assigns dimval instants per scan point to the training and validation set.

        dimval and dimtest are in units of the number of scanpoints.
    """


    subdirs = glob(path)

    test=[]
    train=[]

    for x in subdirs:
        files = glob(x+'*')
        random.shuffle(files)
        test=np.append(test,files[0:dimtest])
        train=np.append(train,files[dimtest:dimtest+dimtrain])



    return list(test),list(train)


class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())

    def standard_deviation_mean(self):
        return self.standard_deviation() / np.sqrt(self.n)



def sort_data(xdata, ydata):

    #use Welford's algorithm to update mean and variance/stddev online
    rs_dict ={}

    master_dict={}
    count_dict={}

    out=[]

    dtype=[('V1_orig', float), ('V1_pred', float), ('V1_std', float)]

    for j, x in enumerate(xdata):
        current_key = x[0]
        if current_key in master_dict:
            master_dict[current_key]+=1*ydata[j]
            count_dict[current_key]+=1


            ## online mean and std
            (rs_dict[current_key]).push(1.0*ydata[j])


        else:
            master_dict[current_key]=1*ydata[j]
            count_dict[current_key]=1

            ## online mean and std
            rs_dict[current_key]= RunningStats()
            (rs_dict[current_key]).push(1.0*ydata[j])


    for current_key in master_dict:

        pred = rs_dict[current_key].mean()
        stdv = rs_dict[current_key].standard_deviation_mean()

        out.append( np.array([(current_key, pred, stdv)],dtype=dtype))


    out=np.stack(out)
    out=np.sort(out, order=['V1_orig', 'V1_pred', 'V1_std'], axis=0)


    return out.view(float)[:,0], out.view(float)[:,1], out.view(float)[:,2]
