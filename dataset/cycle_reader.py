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


class CycleDataset(Dataset):
    def __init__(self, args, frames_path, labels_path, frame_size, sequence_length, 
                 setname='train', random_pad_sample=True, pad_option='default', 
                 uniform_frame_sample=True, random_start_position=True, max_interval=7, random_interval=True):
        self.args = args
        self.dataset = args.dataset
        self.sequence_length = sequence_length

        # multi-modal
        self.multi_modal = args.multi_modal
        self.use_depth = args.use_depth
        self.use_pose = args.use_pose
        self.use_depth = args.use_depth

        # pad option => using for _add_pads function
        self.random_pad_sample = random_pad_sample
        assert pad_option in ['default', 'autoaugment'], "'{}' is not valid pad option.".format(pad_option)
        self.pad_option = pad_option

        # frame sampler option => using for _frame_sampler function
        self.uniform_frame_sample = uniform_frame_sample
        self.random_start_position = random_start_position
        self.max_interval = max_interval
        self.random_interval = random_interval
        
        assert setname in ['train', 'test', 'val'], "'{}' is not valid setname.".format(setname)
        self.data_paths = []
        
        if setname == 'train':
            self.num_classes = 64
        elif setname == 'val':
            self.num_classes = 12
        else:
            self.num_classes = 24

        # this value will using for CategoriesSampler class
        self.classes = [] # ex. [1, 1, 1, ..., 61, 61, 61]

        self.labels = {}
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
                self.labels[action] = int(label) - 1
            csv.close()
            self.num_classes = len(self.labels)
        else:
            self.vid_to_cls = dict()
            with open(args.class_split_folder+'/%s_class.txt'%setname, 'r') as f:
                class_folders = [clss.strip() for clss in f.readlines()]
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
                    self.labels[class_folder] = clss_id
                    self.clss_dict[clss_id].append(vnum)
                    vnum += 1
            self.num_classes = len(class_folders)

        # select normalize value 
        if "resnet" in args.backbone:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.backbone == "r2plus1d":
            normalize = transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
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
        else:
            # transformer in validation or testing phase
            self.transform = transforms.Compose([
                transforms.Resize((frame_size, frame_size)),
                transforms.ToTensor(),
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
            datas = [self.transform_autoaugment(Image.open(sorted_frames_path[s])) for s in sequence]
        else:
            datas = [self.transform(Image.open(sorted_frames_path[s])) for s in sequence]

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
        datas = [self.transform(Image.open(sorted_frames_path[s])) for s in sequence]
        return datas, sequence

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

        cycle_datas, cycle_aux = [], {}

        for indice in indices:
            # get frames and sort
            data_path = self.data_paths[indice]
            sorted_frames_path = sorted(glob.glob(data_path+"/*.jpg"), key=lambda path: int(path.split(".jpg")[0].split("\\" if os.name == 'nt' else "/")[-1]))
            sorted_frames_length = len(sorted_frames_path)
            assert sorted_frames_length != 0, "'{}' Path is not exist or no frames in there.".format(data_path)

            if sorted_frames_length < self.sequence_length:
                datas, seq_ids = self._add_pads(sorted_frames_path)
            else:
                datas, seq_ids = self._frame_sampler(sorted_frames_path)
            datas = torch.stack(datas)

            if self.multi_modal:
                if self.use_depth:
                    if 'depth' not in cycle_aux:
                        cycle_aux['depth'] = []
                    depth_imgs = torch.stack(self.get_modal_imgs(sorted_frames_path, seq_ids, 'depth'))
                    cycle_aux['depth'].append(depth_imgs)
                if self.use_pose:
                    if 'pose' not in cycle_aux:
                        cycle_aux['pose'] = []
                    pose_imgs = torch.stack(self.get_modal_imgs(sorted_frames_path, seq_ids, 'pose'))
                    cycle_aux['pose'].append(pose_imgs)
                if self.use_flow:
                    flow_imgs = torch.stack(self.get_modal_imgs(sorted_frames_path, seq_ids, 'flow'))
                    cycle_aux['flow'].append(flow_imgs)

            if self.dataset == 'ucf101':
                label = self.labels[data_path.split("_")[-3]]
            else:
                #label = self.labels[self.vid_to_cls[ data_path.split('/')[-1][:11] ]]
                label = self.classes[indice]
            cycle_datas.append(datas)
 
        # merge cycle pair
        cycle_datas = torch.stack(cycle_datas)
        for modal in cycle_aux:
            cycle_aux[modal] = torch.stack(cycle_aux[modal])

        return cycle_datas, cycle_aux, label, 


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone().mul(alpha.view(1, 3).expand(3, 3)).mul(self.eigval.view(1, 3).expand(3, 3)).sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))



"""
self.indices = []
labels = np.array(self.classes)
for i in range(1, max(labels)+1):
    index = np.argwhere(labels == i).reshape(-1)
    index = torch.from_numpy(index)
    self.indices.append(index)
"""