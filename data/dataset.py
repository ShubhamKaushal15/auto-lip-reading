# encoding: utf-8
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
import torch
import glob
import editdistance
import random
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


class LipReadSet(Dataset):
    START_TOKEN = "<"
    STOP_TOKEN = ">"
    START_IDX = 28
    STOP_IDX = 29
    letters = ' abcdefghijklmnopqrstuvwxyz' + START_TOKEN + STOP_TOKEN
    start = 1
    """
    start: amount by which to offset integer labels. 0 is usually for blank in CTC
    """

    def __init__(self, video_path, anno_path, file_list, vid_pad = 75, txt_pad = 32, phase = 'train'):
        """
        video_path: directory with s{i} folders. video_path/s{i}/abcd1e/0*.png
        anno_path: directory with s{i} folders. anno_path/s{i}/align/abcd1e.align
        file_list: path to file. file has paths of the form s{i}/abcd1e
        txt_pad = len("place ") + len("green ") + len("with ") + len("A ") + len("seven ") + len("please")
                =  6 + 6 + 5 + 2 + 6 + 7 = 32
        """
        self.anno_path = anno_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase
        
        with open(file_list, 'r') as f:
            self.video_paths = [os.path.join(video_path, line.strip()) for line in f.readlines()]
            
        self.data = []
        for path in self.video_paths:
            items = path.split(os.path.sep)            
            self.data.append((path, items[-2], items[-1]))
        
                
    def __getitem__(self, idx):
        (path, spk, name) = self.data[idx]
        vid = self._load_vid(path)
        anno = self._load_anno(os.path.join(self.anno_path, spk, 'align', name + '.align'))

        # TODO: add data augmentation for training: albumentations
        if(self.phase == 'train'):
            vid = LipReadSet.HorizontalFlip(vid)
          
        vid = LipReadSet.ColorNormalize(vid)
              
        vid_len = vid.shape[0]
        anno_len = anno.shape[0]
        vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
        
        return {'vid': torch.FloatTensor(vid.transpose(3, 0, 1, 2)), # (C, T, H, W)
            'txt': torch.LongTensor(anno),
            'txt_len': anno_len,
            'vid_len': vid_len}
            
    def __len__(self):
        return len(self.data)
        
    def _load_vid(self, p): 
        files = os.listdir(p)
        files = list(filter(lambda file: file.find('.png') != -1, files))
        files = sorted(files, key=lambda file: os.path.splitext(file)[0])
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        array = [cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in array]
        array = np.stack(array, axis=0).astype(np.float32)
        return array
    
    def _load_anno(self, name):
        with open(name, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.lower() in ['sil', 'sp'], txt))
        return LipReadSet.txt2arr(' '.join(txt).lower())
    
    def _padding(self, array, length):
        """
        padding along sequence axis with zeros to become equal to `length`
        """
        # array_copy = array.copy()
        size = array[0].shape
        zeros = np.zeros((length - len(array), *size))
        # for _ in range(length - len(array_copy)):
        #     array_copy += np.zeros(size)
        return np.append(array, zeros, axis=0)
    
    @staticmethod
    def HorizontalFlip(batch_img, p=0.5):
        # (T, H, W, C)
        if random.random() > p:
            batch_img = batch_img[:,:,::-1,...]
        return batch_img

    @staticmethod
    def ColorNormalize(batch_img):
        batch_img = batch_img / 255.0
        return batch_img
    
    @staticmethod
    def txt2arr(txt):
        arr = [LipReadSet.letters.index(c) for c in txt]
        
        return np.array(arr) + LipReadSet.start

    @staticmethod  
    def arr2txt(arr):
        txt = []
        for n in arr:
            if(n >= LipReadSet.start):
                txt.append(LipReadSet.letters[n - LipReadSet.start])     
        return ''.join(txt).strip()
    
    @staticmethod
    def ctc_arr2txt(arr):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= LipReadSet.start):                
                if(len(txt) > 0 and txt[-1] == ' ' and LipReadSet.letters[n - LipReadSet.start] == ' '):
                    pass
                else:
                    txt.append(LipReadSet.letters[n - LipReadSet.start])                
            pre = n
        return ''.join(txt).strip()

    @staticmethod
    def ctc_decode(y):
        y = y.argmax(-1)
        return [LipReadSet.ctc_arr2txt(y[_]) for _ in range(y.size(0))]
            
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer)
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return np.array(cer)

    
class EncodingDataset(Dataset):
    def __init__(self, data_folder_path, set_type, start_token='<', stop_token='>'):
        
        self.img_labels = pd.read_csv(f"{data_folder_path}/{set_type}/labels.csv")
        self.img_dir = f"{data_folder_path}/{set_type}/imgs"  
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #28 is start and 29 is stop
        label = self.img_labels.iloc[idx]
        fname = label.iloc[0]
        
        label["data"] = 28

        label = np.append(np.trim_zeros(label.to_numpy()), 29)
        label.resize(34)
        label = torch.from_numpy(label.astype("long"))
        
        image = pd.read_pickle(f"{self.img_dir}/{fname}")
        image = torch.from_numpy(image.to_numpy())
        
        return image, label