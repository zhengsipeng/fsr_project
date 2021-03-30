import os
import time
import glob
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from .autoaugment import ImageNetPolicy


class TwoCropTransform:
    """Create two crops of the same iamge"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class ContrastDataset(Dataset):
    def __init__(self, args, frames_path, labels_path, frame_size, sequence_length, 
                 setname='train', random_pad_sample=True, pad_option='default', 
                 uniform_frame_sample=True, random_start_position=True, max_interval=7, random_interval=True):
        self.args = args
        self.dataset = args.dataset
        self.sequence_length = sequence_length

        # pad option => using for _add_pads function
        self.random_pad_sample = random_pad_sample
        assert pad_option in ['default', 'autoaugment'], "'{}' is not valid pad option.".format(pad_option)
        self.pad_option = pad_option

        # frame sampler option => using for _frame_sampler function
        self.uniform_frame_sample = uniform_frame_sample
        self.random_start_position = random_start_position
        self.max_interval = max_interval
        self.random_interval = random_interval
        
        assert setname in ['train', 'test', 'val', 'train_val'], "'{}' is not valid setname.".format(setname)
        self.data_paths = []
        
        if setname == 'train':
            self.num_classes = 64
        elif setname == 'val':
            self.num_classes = 12
        elif setname == 'train_val':
            self.num_classes = 76
        elif setname == 'test':
            self.num_classes = 24

        # this value will using for CategoriesSampler class
        self.classes = [] # ex. [1, 1, 1, ..., 61, 61, 61]

        self.clss2id = {}
        self.id2clss = {}
        self.clss_dict = {}
        for i in range(0, self.num_classes):
            self.clss_dict[i] = []

        if self.dataset == 'ucf101':
            if setname == 'train':
                csv = open(os.path.join(labels_path, 'train.csv'))
            if setname == 'test':
                csv = open(os.path.join(labels_path, 'test.csv'))
            lines = csv.readlines()
            for line in lines:
                label, folder_name = line.rstrip().split(',')
                action = folder_name.split('_')[1]
                self.data_paths.append(os.path.join(frames_path, folder_name))
                self.classes.append(int(label))
                self.clss2id[action] = int(label) - 1
            csv.close()
            self.num_classes = len(self.clss2id)
        else:
            setnames = setname.split('_')
            class_folders = []
            for setname in setnames:
                self.vid_to_cls = dict()
                with open(args.class_split_folder+'/%s_class.txt'%setname, 'r') as f:
                    class_folders += [clss.strip() for clss in f.readlines()]
            
            class_folders.sort()
            self.class_folders = class_folders
            vnum = 0
            
            for clss_id, class_folder in enumerate(self.class_folders):
                video_folders = os.listdir(os.path.join(frames_path, class_folder))
                video_folders.sort()
                for video_folder in video_folders:
                    vid = video_folder[:11]
                    self.data_paths.append(os.path.join(frames_path, class_folder, video_folder))
                    self.classes.append(clss_id)
                    self.vid_to_cls[vid] = class_folder
                    self.clss_dict[clss_id].append(vnum)
                    self.clss2id[class_folder] = clss_id
                    self.id2clss[clss_id] = class_folder
                    vnum += 1
            self.num_classes = len(class_folders)

        # select normalize value 
        if "resnet" in args.backbone:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       
        # transformer in training phase
        if setname == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=frame_size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
            #self.transform = TwoCropTransform(transform)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((frame_size, frame_size)),
                transforms.ToTensor(),
                normalize,
            ])
        # autoaugment transformer for insufficient frames in training phase
        self.transform_autoaugment = transforms.Compose([
            transforms.RandomResizedCrop(size=frame_size, scale=(0.2, 1.)),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.data_paths)

    def _add_pads(self, sorted_frames_path):
        # get sorted frames length to list
        sequence = np.arange(len(sorted_frames_path))
        
        if self.random_pad_sample:
            # random sampling of pad
            add_sequence = np.random.choice(sequence, self.sequence_length - len(sequence))
        else:
            # repeated of first pad
            add_sequence = np.repeat(sequence[0], self.sequence_length - len(sequence))
        
        # sorting the pads
        sequence = sorted(np.append(sequence, add_sequence, axis=0))

        # transform to Tensor
        if self.pad_option == 'autoaugment':
            datas = []
            for s in sequence:
                img = Image.open(sorted_frames_path[s])
                datas.append(self.transform_autoaugment(img))
        else:
            datas = []
            imgs = []
            for s in sequence:
                img = Image.open(sorted_frames_path[s])
                imgs.append(img)
            _datas = []
            for s in range(len(sequence)):
                _datas.append(self.transform(imgs[s]))
            datas.append(torch.stack(_datas))

            _datas = []
            for s in range(len(sequence)):
                _datas.append(self.transform(imgs[s]))
            datas.append(torch.stack(_datas))

        return datas, sequence

    def _frame_sampler(self, sorted_frames_path):
        # get sorted frames length to list
        sorted_frames_length = len(sorted_frames_path)

        # set a sampling strategy
        if self.uniform_frame_sample:
            # set a default interval
            interval = (sorted_frames_length // self.sequence_length) - 1
            if self.max_interval != -1 and interval > self.max_interval:
                interval = self.max_interval

            # set a interval with randomly
            if self.random_interval:
                interval = np.random.permutation(np.arange(start=0, stop=interval + 1))[0]
    
            # get a require frames
            require_frames = ((interval + 1) * self.sequence_length - interval)
            
            # get a range of start position
            range_of_start_position = sorted_frames_length - require_frames

            # set a start position
            if self.random_start_position:
                start_position = np.random.randint(0, range_of_start_position + 1)
            else:
                start_position = 0
            
            sequence = list(range(start_position, require_frames + start_position, interval + 1))
        else:
            sequence = sorted(np.random.permutation(np.arange(sorted_frames_length))[:self.sequence_length])

        # transform to Tensor
        datas = []
        '''
        for s in sequence:
            img = Image.open(sorted_frames_path[s])
            datas.append(self.transform(img))
        '''
        imgs = []
        for s in sequence:
            img = Image.open(sorted_frames_path[s])
            imgs.append(img)
        _datas = []
        for s in range(len(sequence)):
            _datas.append(self.transform(imgs[s]))
        datas.append(torch.stack(_datas))

        _datas = []
        for s in range(len(sequence)):
            _datas.append(self.transform(imgs[s]))
        datas.append(torch.stack(_datas))

        return datas, sequence

    def __getitem__(self, index):
        i = index % len(self.data_paths)

        # get frames and sort
        data_path = self.data_paths[i]
        sorted_frames_path = sorted(glob.glob(data_path+"/*.jpg"), key=lambda path: int(path.split(".jpg")[0].split("\\" if os.name == 'nt' else "/")[-1]))
        sorted_frames_length = len(sorted_frames_path)
        assert sorted_frames_length != 0, "'{}' Path is not exist or no frames in there.".format(data_path)

        if sorted_frames_length < self.sequence_length:
            datas, _ = self._add_pads(sorted_frames_path)
        else:
            datas, _ = self._frame_sampler(sorted_frames_path)

        if self.dataset == 'ucf101':
            label = self.clss2id[data_path.split("_")[-3]]
        else:
            label = self.classes[i]

        return datas, label


