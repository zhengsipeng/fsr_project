from torch.utils.data import DataLoader
from dataset.base_reader import EpisodeSampler
from dataset.data_reader import ContrastDataset, GeneralDataset, CycleDataset


def prepare_dataset(args):
    if args.method == 'supcon':
        train_dataset = GeneralDataset(
            args,
            frames_path=args.frames_path,
            labels_path=args.labels_path,
            frame_size=args.frame_size,
            sequence_length=args.sequence_length,
            setname='train_val',
            random_pad_sample=args.random_pad_sample,
            pad_option=args.pad_option,
            uniform_frame_sample=args.uniform_frame_sample,
            random_start_position=args.random_start_position,
            max_interval=args.max_interval,
            random_interval=args.random_interval,
        )
    elif args.method == 'cycle':
        train_dataset = CycleDataset(
            args,
            frames_path=args.frames_path,
            labels_path=args.labels_path,
            frame_size=args.frame_size,
            sequence_length=args.sequence_length,
            setname='train',
            random_pad_sample=args.random_pad_sample,
            pad_option=args.pad_option,
            uniform_frame_sample=args.uniform_frame_sample,
            random_start_position=args.random_start_position,
            max_interval=args.max_interval,
            random_interval=args.random_interval,
        )
    else:
        train_dataset = GeneralDataset(
            args,
            frames_path=args.frames_path,
            labels_path=args.labels_path,
            frame_size=args.frame_size,
            sequence_length=args.sequence_length,
            setname='train',
            # pad options
            random_pad_sample=args.random_pad_sample,
            pad_option=args.pad_option,
            # frame sample options
            uniform_frame_sample=args.uniform_frame_sample,
            random_start_position=args.random_start_position,
            max_interval=args.max_interval,
            random_interval=args.random_interval,
        )
    val_dataset = GeneralDataset(
        args,
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        setname='test',
        random_pad_sample=False,
        pad_option='default',
        uniform_frame_sample=True,
        random_start_position=False,
        max_interval=7,
        random_interval=False,
    )
    if args.meta_learn:
        print("[train] number of videos / classes: {} / {}, [val] number of videos / classes: {} / {}".format(len(train_dataset), train_dataset.num_classes, len(val_dataset), val_dataset.num_classes))
        print("total training episodes: {}".format(args.num_epochs * args.num_train_episode))
        train_sampler = EpisodeSampler(train_dataset.classes, args.num_train_episode, args.way, args.shot, args.query)
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, sampler=None)    

    val_sampler = EpisodeSampler(val_dataset.classes, args.num_val_episode, args.way, args.shot, args.query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

    id2clss = train_dataset.id2clss

    return train_loader, val_loader, id2clss