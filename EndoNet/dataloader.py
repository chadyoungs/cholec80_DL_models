from doctest import OutputChecker
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
import glob
import joblib

from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
import cv2

from utils import TrainTestOptError
import config

# train-set-info and test-set-info
train_test_sets_info_file_loc = os.path.join( os.path.dirname(os.path.abspath(__file__)), "dataset")


class Dataset(Dataset):
    def __init__(self, train_test_opt):
        """
        train_test_opt should be one of "train" and "test"
        """
        self.train_test_opt = train_test_opt
        
        # whether to use the memory which could speed up the training process
        self.use_mem = False
        # saving the imgs in memory
        self.imgs = []

        # data and label
        self.data_list, self.phase_label_list, self.tool_label_list = self._build_data_label()
           
    def _build_data_label(self):
        '''
        build the dataset
        '''
        if self.train_test_opt not in ["train", "test"]:
            raise TrainTestOptError("Check the train test option setting")
        
        set_info_csv_loc = os.path.join(train_test_sets_info_file_loc, self.train_test_opt+"_set_info.csv")
        set_info_csv = pd.read_csv(set_info_csv_loc)
        
        return np.asarray(set_info_csv["file_loc"]), np.asarray(set_info_csv[config.phase_label_in_csv]).astype(np.float32), np.asarray(set_info_csv[config.tool_labels_without_NoTool]).astype(np.float32)
    
    def transform(self, img):
        return transforms.Compose([
            transforms.Resize((config.img_w, config.img_h)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])(img)

    def _read_img_and_transform(self, img):
        crop_img = Image.open(img).crop((50, 0, 804, 480)).convert('RGB')
        
        return self.transform(crop_img)
    
    def __getitem__(self, index):
        # reading the imgs
        if self.use_mem:
            X = self.imgs[index]
        else:
            X = self._read_img_and_transform(self.data_list[index])
        
        # transform to tensor
        y_phase = self.phase_label_list[index]
        y_label = self.tool_label_list[index]
        
        return X, y_label, y_phase

    def __len__(self):
            
        return len(self.data_list)
