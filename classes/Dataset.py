import torch
import numpy as np


'Pytorch dataset class - describes OPG'
class Dataset(torch.utils.data.Dataset):

    def __init__(self, ids, path_to_data):
        
        'Initialization'
        self.ids = ids #age_ourid_name; folder structure is folder_ 11_ 987654_12345.dcm_ 12345.json ; .png
        self.path_to_data = path_to_data

    def __len__(self):

        'Denotes the total number of samples'
        return len(self.ids)

    def __getitem__(self, index):
        
        # Generate one sample of data
        sample = {}

        # Get paths from id
        id = self.ids[index]
       
        # Print sample id
        print(f'Sample: {id}')

        # Get data from disk
        opg = Image.open(path_image)

        # Class label
 
        return sample
        
