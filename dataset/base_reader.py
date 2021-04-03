import os
import time
import glob
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, frames_path, labels_path, frame_size, sequence_length, 
                 setname='train', random_pad_sample=True, pad_option='default', 
                 uniform_frame_sample=True, random_start_position=True, max_interval=7, random_interval=True):
        self.args = args
        self.dataset = args.dataset
        self.sequence_length = sequence_length

        # multi-modal
        self.use_depth = args.use_depth
        self.use_pose = args.use_pose
        self.use_flow = args.use_flow
        if self.use_depth or self.use_pose or self.use_flow:
            self.multi_modal = True

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
                with open(args.class_split_folder+'/%s_class.txt'%setname, 'r') as f:
                    class_folders += [clss.strip() for clss in f.readlines()]
            
            class_folders.sort()
            self.class_folders = class_folders
            vnum = 0
            
            for clss_id, class_folder in enumerate(self.class_folders):
                video_folders = os.listdir(os.path.join(frames_path, class_folder))
                video_folders.sort()
                for video_folder in video_folders:
                    self.data_paths.append(os.path.join(frames_path, class_folder, video_folder))
                    self.classes.append(clss_id)
                    self.clss_dict[clss_id].append(vnum)
                    self.clss2id[class_folder] = clss_id
                    self.id2clss[clss_id] = class_folder
                    vnum += 1
            self.num_classes = len(class_folders)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if setname == 'train':
            self.vis_transform = transforms.Compose([
                #transforms.Resize((frame_size + 16, frame_size + 59)),
                #transforms.CenterCrop((frame_size, frame_size)),
                transforms.ToTensor(),
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
            transforms.RandomResizedCrop(size=frame_size, scale=(0.2, 1.)),
            ImageNetPolicy(),
            transforms.ToTensor(),
            normalize,
        ])

    def get_modal_imgs(self, frame_paths, sequence_ids, modal):
        modal_paths = []
        for frame_path in frame_paths:
            if modal == 'depth':
                _path = 'data/multi_modal/depth/greyscale/'+frame_path[12:]  # I think grey scale is better
            if modal == 'pose':
                raise NotImplementedError
            if modal == 'flow':
                raise NotImplementedError
            modal_paths.append(_path)
        datas = [self.transform(Image.open(modal_paths[s]).convert('RGB')) for s in sequence_ids]
        
        return datas

    def add_pads_sequence(self, sorted_frames_path):
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
        
        return sequence

    def sequence_sampler(self, sorted_frames_path):
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

        return sequence
    
    def get_iter_data(self, i):
        # get frames and sort
        data_path = self.data_paths[i]
        sorted_frames_path = sorted(glob.glob(data_path+"/*.jpg"), key=lambda path: int(path.split(".jpg")[0].split("\\" if os.name == 'nt' else "/")[-1]))
        sorted_frames_length = len(sorted_frames_path)
        assert sorted_frames_length != 0, "'{}' Path is not exist or no frames in there.".format(data_path)

        # we may be encounter that error such as
        # 1. when insufficient frames of video rather than setted sequence length, _add_pads function will be solve this problem
        # 2. when video has too many frames rather than setted sequence length, _frame_sampler function will be solve this problem
        if sorted_frames_length < self.sequence_length:
            datas, seq_ids, frames = self._add_pads(sorted_frames_path)
        else:
            datas, seq_ids, frames = self._frame_sampler(sorted_frames_path)
        
        if self.args.meta_learn:
            datas = torch.stack(datas)

        modal_aux = dict()
        if self.use_depth:
            depth_imgs = torch.stack(self.get_modal_imgs(sorted_frames_path, seq_ids, 'depth'))
            modal_aux['depth'] = depth_imgs
            #print(depth_imgs.shape)
        if self.use_pose:
            pose_imgs = torch.stack(self.get_modal_imgs(sorted_frames_path, seq_ids, 'pose'))
            modal_aux['pose'] = pose_imgs
        if self.use_flow:
            flow_imgs = torch.stack(self.get_modal_imgs(sorted_frames_path, seq_ids, 'flow'))
            modal_aux['flow'] = flow_imgs
        
        if self.dataset == 'ucf101':
            label = self.clss2id[data_path.split("_")[-3]]
        else:
            label = self.classes[i]
   
        return datas, modal_aux, label, frames


# original code: https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
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


class EpisodeSampler():
    """
    Episode Generation
    """
    def __init__(self, labels, num_episode, way, shot, query):
        self.num_episode = num_episode
        self.way = way
        self.shot = shot
        self.query = query
        self.shots = shot + query

        labels = np.array(labels)
        self.indices = []
        for i in range(1, max(labels) + 1):
            index = np.argwhere(labels == i).reshape(-1)
            index = torch.from_numpy(index)
            self.indices.append(index)

    def __len__(self):
        return self.num_episode
    
    def __iter__(self):
        for i in range(self.num_episode):
            batchs = []
            classes = torch.randperm(len(self.indices))[:self.way] # bootstrap(class)
            for c in classes:
                l = self.indices[c]
                pos = torch.randperm(len(l))[:self.shots] # bootstrap(shots)
                batchs.append(l[pos])
            batchs = torch.stack(batchs).t().reshape(-1)
            yield batchs

"""
# way*(shot+query) -> way*shot+way*query
#batchs = batchs.reshape(self.way, self.shots)
#shot = batchs[:, :self.shot].reshape(-1)
#query = batchs[:, self.shot:].reshape(-1)
#batchs = torch.cat([shot, query])
"""


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img

