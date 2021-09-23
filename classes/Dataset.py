import torch
import numpy as np


'Pytorch dataset class - describes OPG'
class Dataset(torch.utils.data.Dataset):

    def __init__(self, ids, path_to_data):
        
        'Initialization with list of ids'
        self.ids = ids # all ids (row containing one simulation instance)
        self.path_to_data = path_to_data # path to where to get the datafrom 

    def __len__(self):

        'Denotes the total number of samples'
        return len(self.ids)

    def __getitem__(self, index):
       
        # Print sample id
        print(f'Sample: {index}')

        # Get data from disk
        data = #read from file, maybe already in init to reduce access time and hold in memory +
        X_data = torch.zeros(size=(self.backtime, N*(L-(self.backtime+self.foretime)), self.input_size-5))
        X_PP_data = torch.zeros(size=(self.backtime, N*(L-(self.backtime+self.foretime)), 5))
        y_data = torch.zeros(size=(self.foretime, N*(L-(self.backtime+self.foretime)), self.input_size-5))
        
        for i in range(L-(self.backtime+self.foretime)):
            X = data[i:i+self.backtime]
            X_PP = data[i:i+self.backtime] 
            y = data[i+self.backtime:i+self.backtime+self.foretime]
            X_data[:, i*N:(i+1)*N, :] = X
            X_PP_data[:, i*N:(i+1)*N, :] = X_PP
            y_data[:, i*N:(i+1)*N, :] = y

        # sample = data[index]
        # Split data into many slices


        
 
        return sample
        
