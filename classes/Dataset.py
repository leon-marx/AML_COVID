import torch
import numpy as np


'Pytorch dataset class - describes OPG'
class Dataset(torch.utils.data.Dataset):

    def __init__(self, path_to_data, path_to_PP_data, indices, backtime, foretime):
        """
        path_to_data: the path, where the dataset is stored. The dataset should be stored with shape (L, N, 1)
        path_to_PP_data: the path, where the PP data is stored. The dataset should be stored with shape (L, N, 5)
        indices: array of indices to pick as the dataset. This is used in order to differentiate between test and train set.
        """
        self.data = torch.load(path_to_data)[:, indices, :]
        self.PP_data = torch.load(path_to_PP_data)[:, indices, :]

        self.X_data = torch.zeros(size=(backtime, self.data.shape[1]*(self.data.shape[0]-(backtime+foretime)), 1))
        self.X_PP_data = torch.zeros(size=(backtime, self.data.shape[1]*(self.data.shape[0]-(backtime+foretime)), 5))
        self.y_data = torch.zeros(size=(foretime, self.data.shape[1]*(self.data.shape[0]-(backtime+foretime)), 1))
        
        for i in range(self.data.shape[0]-(backtime+foretime)):
            self.X_data[:, i*self.data.shape[1]:(i+1)*self.data.shape[1], :] = self.data[i:i+backtime]
            self.X_PP_data[:, i*self.data.shape[1]:(i+1)*self.data.shape[1], :] = self.PP_data[i:i+backtime]
            self.y_data[:, i*self.data.shape[1]:(i+1)*self.data.shape[1], :] = self.data[i+backtime:i+backtime+foretime]

    def __len__(self):
        return self.X_data.shape[1]

    def __getitem__(self, index):
        return self.X_data[:, index, :], self.X_PP_data[:, index, :], self.y_data[:, index, :]
