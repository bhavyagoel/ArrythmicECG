import os
import numpy as np
import scipy.io
import scipy.io
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def mat_reader(path_file, square):
    ecg = scipy.io.loadmat(path_file)
    ecg = {k: v for k, v in ecg.items() if k[0] != '_'}
    if square:
        return (ecg['val']).reshape(1, 60, 60)
    else:
        return (ecg['val']).reshape(1, 3600)


def x_y_data(data):
    np.random.shuffle(data)
    np.random.shuffle(data)

    y = []
    x = []
    for i in range(len(data)):
        k = data[i][0]
        y.append(k)
        m = data[i][1]
        x.append(torch.tensor(m).to(device))
    # print(y)
    y = torch.tensor(y).type(torch.cuda.LongTensor)
    x = torch.stack(x)

    return x, y


def data_loader(path_dir):
    data = []

    for _class, _cname in enumerate(os.listdir(path_dir)):
        frags_dir = os.path.join(path_dir, _cname)
        count = 0
        for _frags in os.listdir(frags_dir):
            frags_loc = os.path.join(frags_dir, _frags)
            x = mat_reader(frags_loc, False)
            y = _class
            count += 1
            data.append([x, y])

        if len(os.listdir(frags_dir)) < 284:
            inc_count = 284 - len(os.listdir(frags_dir))

            each_file = int(inc_count / len(os.listdir(frags_dir)))
            for _frags in os.listdir(frags_dir):
                for i in range(each_file):
                    count += 1
                    if i % 2 == 0:
                        alpha = np.copy(x)
                        alpha = np.roll(alpha, 2 * i)
                        y = _class
                        data.append([alpha, y])
                    else:
                        alpha = np.copy(x)
                        alpha = np.roll(alpha, -2 * i)
                        y = _class
                        data.append([alpha, y])
            print(_class, _cname, count)

    x, y = x_y_data(data)
    del data

    num_train = len(x)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))

    np.random.shuffle(indices)

    valid_idx, test_idx, train_idx = indices[:split], indices[split:2 * split], indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(TensorDataset(x, y), batch_size=64, sampler=train_sampler)
    valid_loader = DataLoader(TensorDataset(x, y), batch_size=64, sampler=valid_sampler)
    test_loader = DataLoader(TensorDataset(x, y), batch_size=64, sampler=test_sampler)

    return train_loader, test_loader, valid_loader
