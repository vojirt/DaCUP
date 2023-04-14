import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.datasets import cityscapes_2class, bdd100k_2class 
from mypath import Path



class CityBDD100k_2class(data.Dataset):
    NUM_CLASSES = 2 

    def __init__(self, cfg, root, split):
        self.split = split
        self.files = {}

        self.train_set_city = cityscapes_2class.CityscapesSegmentation_2Class(cfg, root=Path.dataset_root_dir("cityscapes_2class"), split=self.split)
        self.train_set_bdd = bdd100k_2class.BDD100k_2class(cfg, root=Path.dataset_root_dir("bdd100k_2class"), split=self.split)

        if cfg.DATASET.FULL_DATA:
            self.num_data = len(self.train_set_city)+len(self.train_set_bdd)
            self.idd = np.random.permutation(self.num_data)  
            self.db_mapping_city = dict(zip(self.idd[:len(self.train_set_city)], np.arange(len(self.train_set_city))))
            self.db_mapping_bdd = dict(zip(self.idd[len(self.train_set_city):], np.arange(len(self.train_set_bdd))))
        else:
            len_city = len(self.train_set_city)
            len_bdd = len(self.train_set_bdd)
            slen_city = int(len_city/2) 
            slen_bdd = int(len_bdd/6) 
            self.num_data = slen_city + slen_bdd 
            self.idd = np.random.permutation(self.num_data)  
            self.db_mapping_city = dict(zip(self.idd[:slen_city], np.random.permutation(len(self.train_set_city))[:slen_city]  ))
            self.db_mapping_bdd = dict(zip(self.idd[slen_city:], np.random.permutation(len(self.train_set_bdd))[:slen_bdd]  ))

    
        if self.num_data == 0:
            raise Exception("No files for split=[%s] found in %s" % (split,Path.dataset_root_dir("bdd100k_2class") + ";" +Path.dataset_root_dir("cityscapes_2class")))

        print("(City+BDD100k) Total Found %d %s images" % (self.num_data, split))

    def __len__(self):
        return self.num_data 

    def __getitem__(self, index):
        id_city = self.db_mapping_city.get(index, -1)
        id_bdd = self.db_mapping_bdd.get(index, -1)
        if id_city == -1 and id_bdd > -1:
            return self.train_set_bdd[id_bdd]
        elif id_city > -1 and id_bdd == -1:
            return self.train_set_city[id_city]
        else:
            # shouldn't ever happen :)
            assert False, "Error in indexing of CityBDD100k_2class id_city={}, id_bdd={}".format(id_city, id_bdd)


        
        

