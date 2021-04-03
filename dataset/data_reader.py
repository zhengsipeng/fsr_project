import os
import time
import glob
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from .base_reader import ImageNetPolicy, Lighting, EpisodeSampler, BaseDataset


# ==================
# Ceneral Datastet
# ==================
class GeneralDataset(BaseDataset):
    def __init__(self, args, frames_path, labels_path, frame_size, sequence_length, 
                 setname='train', random_pad_sample=True, pad_option='default', 
                 uniform_frame_sample=True, random_start_position=True, max_interval=7, random_interval=True):
        super(GeneralDataset, self).__init__(args, frames_path, labels_path, frame_size, sequence_length, setname, random_pad_sample, pad_option, 
                                              uniform_frame_sample, random_start_position, max_interval, random_interval)
    
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # transformer in training phase
        if setname == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((frame_size + 16, frame_size + 59)),
                transforms.CenterCrop((frame_size, frame_size)),
                
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4
                ),
                transforms.ToTensor(),
                Lighting(alphastd=0.1, eigval=[0.2175, 0.0188, 0.0045],
                                        eigvec=[[-0.5675, 0.7192, 0.4009],
                                                [-0.5808, -0.0045, -0.8140],
                                                [-0.5836, -0.6948, 0.4203]]
                ),
                normalize,
            ])
        
        # autoaugment transformer for insufficient frames in training phase
        self.transform_autoaugment = transforms.Compose([
            transforms.Resize((frame_size + 16, frame_size + 59)),
            transforms.CenterCrop((frame_size, frame_size)),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.data_paths)

    def _add_pads(self, sorted_frames_path):
        sequence = self.add_pads_sequence(sorted_frames_path)

        # transform to Tensor
        if self.pad_option == 'autoaugment':
            datas = [self.transform_autoaugment(Image.open(sorted_frames_path[s])) for s in sequence]
        else:
            datas = [self.transform(Image.open(sorted_frames_path[s])) for s in sequence]

        return datas, sequence, None

    def _frame_sampler(self, sorted_frames_path):
        sequence = self.sequence_sampler(sorted_frames_path)

        # transform to Tensor
        datas = [self.transform(Image.open(sorted_frames_path[s])) for s in sequence]
    
        return datas, sequence, None

    def __getitem__(self, index):
        datas, modal_aux, label, _ = self.get_iter_data(index % len(self))
        
        return datas, label


# ==================
# Contrast Datastet
# ==================
class ContrastDataset(BaseDataset):
    def __init__(self, args, frames_path, labels_path, frame_size, sequence_length, 
                 setname='train', random_pad_sample=True, pad_option='default', 
                 uniform_frame_sample=True, random_start_position=True, max_interval=7, random_interval=True):
        super(ContrastDataset, self).__init__(args, frames_path, labels_path, frame_size, sequence_length, setname, random_pad_sample, pad_option, 
                                              uniform_frame_sample, random_start_position, max_interval, random_interval)
    

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

    def __len__(self):
        return len(self.data_paths)

    def _add_pads(self, sorted_frames_path):
        sequence = self.add_pads_sequence(sorted_frames_path)

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

        return datas, sequence, None

    def _frame_sampler(self, sorted_frames_path):
        sequence = self.sequence_sampler(sorted_frames_path)

        # transform to Tensor
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

        return datas, sequence, None

    def __getitem__(self, index):
        datas, _, label, _ = self.get_iter_data(index % len(self))
        
        return datas, label


# ==================
# Cycle Datastet
# ==================
class CycleDataset(BaseDataset):
    def __init__(self, args, frames_path, labels_path, frame_size, sequence_length, 
                 setname='train', random_pad_sample=True, pad_option='default', 
                 uniform_frame_sample=True, random_start_position=True, max_interval=7, random_interval=True):
        super(CycleDataset, self).__init__(args, frames_path, labels_path, frame_size, sequence_length, setname, random_pad_sample, pad_option, 
                                              uniform_frame_sample, random_start_position, max_interval, random_interval)
    
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if setname == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((frame_size + 16, frame_size + 59)),
                transforms.CenterCrop((frame_size, frame_size)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4
                ),
                transforms.ToTensor(),
                Lighting(alphastd=0.1, eigval=[0.2175, 0.0188, 0.0045],
                                        eigvec=[[-0.5675, 0.7192, 0.4009],
                                                [-0.5808, -0.0045, -0.8140],
                                                [-0.5836, -0.6948, 0.4203]]
                ),
                normalize,
            ])

        # autoaugment transformer for insufficient frames in training phase
        self.transform_autoaugment = transforms.Compose([
            transforms.Resize((frame_size + 16, frame_size + 59)),
            transforms.CenterCrop((frame_size, frame_size)),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ])

        # cycle consistency option
        self.iter_reset = args.iter_reset
        self.make_cycle_indexes()

    def __len__(self):
        return self.args.epoch_iter

    def _add_pads(self, sorted_frames_path):
        sequence = self.add_pads_sequence(sorted_frames_path)

        # transform to Tensor
        if self.pad_option == 'autoaugment':
            datas, frames = [], []
            for s in sequence:
                img = Image.open(sorted_frames_path[s])
                datas.append(self.transform_autoaugment(img))
                frames.append(self.vis_transform(img))
        else:
            datas, frames = [], []
            for s in sequence:
                img = Image.open(sorted_frames_path[s])
                datas.append(self.transform(img))
                frames.append(self.vis_transform(img))

        return datas, sequence, frames

    def _frame_sampler(self, sorted_frames_path):
        sequence = self.sequence_sampler(sorted_frames_path)

        # transform to Tensor
        datas, frames = [], []
        for s in sequence:
            img = Image.open(sorted_frames_path[s])
            datas.append(self.transform(img))
            frames.append(self.vis_transform(img))
            
        return datas, sequence, frames

    def make_cycle_indexes(self):
        """
        make or reset cycle indexes
        """
        self.cycle_indexes = []
        for i in range(self.iter_reset):
            clss = random.randint(0, self.num_classes-1)
            indexes = self.clss_dict[clss]  # 100 videos per class
            rand_idxs = random.sample(range(100), 2)
            idx1, idx2 = indexes[rand_idxs[0]], indexes[rand_idxs[1]]
            self.cycle_indexes.append([idx1, idx2])

    def __getitem__(self, index):
        # generate the cycle indexes for next "iter_reset" iterations
        if (index+1) % self.iter_reset == 0:
            self.make_cycle_indexes()

        # index is the cycle index 
        # we randomly select two videos from the same video
        indices = self.cycle_indexes[index%self.iter_reset]  # 2-D vector

        cycle_datas, cycle_aux = [], {'depth': [], 'pose': [], 'flow': []}
        cycle_frames = []

        for i in indices:
            datas, modal_aux, label, frames = self.get_iter_data(i)

            if self.use_depth:
                cycle_aux['depth'].append(modal_aux['depth'])
            if self.use_pose:
                cycle_aux['pose'].append(modal_aux['pose'])
            if self.use_flow:
                cycle_aux['depth'].append(modal_aux['flow'])

            cycle_datas.append(datas)
            cycle_frames.append(frames)
 
        # merge cycle pair
        cycle_datas = torch.stack(cycle_datas)
        for modal in cycle_aux:
            cycle_aux[modal] = torch.stack(cycle_aux[modal])

        return cycle_datas, cycle_aux, label, frames
