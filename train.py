import torch
import os
import sys
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from nets.protonet import ProtoNet
from nets.trx import TRX
from utils.utils import path_check, args_print_save, printer
from utils.loss import cross_entropy_trx
from dataset.general_reader import GeneralDataset, EpisodeSampler
from configs.add_args import add_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ===========================Basic options===================================
    parser.add_argument("--dataset", type=str, default="kinetics100")
    parser.add_argument("--class_split_folder", type=str, default="data/splits/kinetics100/")
    parser.add_argument("--frames_path", type=str, default="data/images/ucf101/")
    parser.add_argument("--labels_path", type=str, default="data/splits/ucf101/")
    parser.add_argument("--save_path", type=str, default="ckpt/train1/")
    parser.add_argument("--tensorboard_path", type=str, default="./tensorboard/train1")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--episode_per_batch", type=int, default=8)
    parser.add_argument("--grad_setting", type=str, default='basic')
    # ===========================Few-shot options=================================
    parser.add_argument("--model", type=str, default="proto")  # proto, trx
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--num_train_episode", type=int, default=7000)
    parser.add_argument("--num_val_episode", type=int, default=3000)
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--metric", type=str, default="cosine")  # euclidean, relation, cosine
    # ===========================Multi-modal options==============================
    parser.add_argument("--multi_modal", type=bool, default=True)
    parser.add_argument("--use_depth", type=bool, default=True)
    parser.add_argument("--use_pose", type=bool, default=False)
    parser.add_argument("--use_flow", type=bool, default=False)
    parser.add_argument("--fusion", type=str, default=None)  # None, "cen"
    parser.add_argument("--sharing", action="store_true")
    # ===========================Dataset options==================================
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--sequence_length", type=int, default=8)
    # pad options
    parser.add_argument("--random_pad_sample", action="store_true")
    parser.add_argument("--pad_option", type=str, default="default")
    # frame options
    parser.add_argument("--uniform_frame_sample", action="store_true")
    parser.add_argument("--random_start_position", action="store_true")
    parser.add_argument("--max_interval", type=int, default=64)
    parser.add_argument("--random_interval", action="store_true")
    # ===========================Backbone options=================================
    parser.add_argument("--backbone", type=str, default="resnet18")  # resnet18, resnet34, resnet50
    parser.add_argument("--freeze_all", type=bool, default=False)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scheduler_step_size", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.9)
    parser.add_argument("--bn_threshold", type=float, default=2e-2)    
    args = add_args(parser)

    # check options
    assert args.model in ["proto", "trx"], "'{}' model is invalid.".format(args.model)
    assert args.backbone in ["resnet18", "resnet34", "resnet50", "r2plus1d"], "'{}' backbone is invalid.".format(args.model)
    assert args.metric in ["cosine", "euclidean", "relation"], "'{}' metric is invalid.".format(args.metric)

    # path to save
    #path_check(args.save_path)
    
    # path to tensorboard
    writer = SummaryWriter(args.tensorboard_path)
    
    # print args and save it in the save_path
    args_print_save(args)

    # ==========
    # Dataset
    # ==========
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
    
    # do not use the autoaugment on the validation or test dataset
    val_dataset = GeneralDataset(
        args,
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        setname='val',
        # pad options
        random_pad_sample=False,
        pad_option='default',
        # frame sample options
        uniform_frame_sample=True,
        random_start_position=False,
        max_interval=7,
        random_interval=False,
    )
    print("[train] number of videos / classes: {} / {}, [val] number of videos / classes: {} / {}".format(len(train_dataset), train_dataset.num_classes, len(val_dataset), val_dataset.num_classes))
    print("total training episodes: {}".format(args.num_epochs * args.num_train_episode))
    train_sampler = EpisodeSampler(train_dataset.classes, args.num_train_episode, args.way, args.shot, args.query)
    val_sampler = EpisodeSampler(val_dataset.classes, args.num_val_episode, args.way, args.shot, args.query)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=0 if os.name == 'nt' else 4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=0 if os.name == 'nt' else 4, pin_memory=True)

    # ============
    # Model Build
    # ============
    # select a model, i prepaired two models for simple testing
    if args.model == "proto":
        model = ProtoNet(
            args,
            way=args.way,
            shot=args.shot,
            query=args.query,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            bidirectional=args.bidirectional,
        )
    elif args.model == "trx":
        model = TRX(args,
                    way=args.way,
                    shot=args.shot,
                    query=args.query)
    else:
        raise NotImplementedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if args.num_gpus > 1:
        model.distribute_model(args.num_gpus)
    
    # =============
    # Optimization
    # =============
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    # ====================
    # Episodic Training
    # ====================
    best = 0 # top1 best accuracy
    total_loss = 0
    total_acc = 0
    n_iter_train = 0
    n_iter_val = 0
    print("train... {}-way {}-shot {}-query".format(args.way, args.shot, args.query))
    for e in range(1, args.num_epochs+1):
        train_acc = []
        train_loss = []
        batch_loss = None
        model.train()
        for i, (datas, modal_aux, _) in enumerate(train_loader):
            # prepare multi-modal auxilary
            aux = dict()
            for modal in ['depth', 'pose', 'flow']:
                if modal in modal_aux:
                    aux[modal] = modal_aux[modal].to(device)
            #print(datas.shape)
            print(aux['depth'][0])
            print(aux['depth'][1])
            print(aux['depth'].shape)
            assert 1==0   
            datas = datas.to(device)  # way*(shot+query), t, c, h, w
            datas = datas.reshape(args.way, args.shot+args.query, args.sequence_length, 3, args.frame_size, args.frame_size)
            assert 1==0
            pivot = args.way * args.shot
            shot, query = datas[:pivot], datas[pivot:]
            support_labels = torch.arange(args.way).repeat(args.shot).to(device)
            labels = torch.arange(args.way).repeat(args.query).to(device)
            if args.model == 'proto':
                pred = model(shot, query, aux)
                loss = F.cross_entropy(pred, labels)
            elif args.model == 'trx':
                pred = model(shot, query, support_labels, aux)     
                loss = cross_entropy_trx(pred, labels, device)
            # calculate loss
            # onehot_labels = Variable(torch.zeros(args.way*args.query, args.way).scatter_(1, torch.arange(args.way).repeat(args.query).view(-1, 1), 1)).to(device) 
            # loss = F.mse_loss(pred, onehot_labels)

            train_loss.append(loss.item())
            total_loss = sum(train_loss)/len(train_loss)

            # update weight
            optimizer.zero_grad()
            if args.grad_setting == 'basic':
                loss.backward()
            elif args.grad_setting == 'batch_grad':
                if  i % args.episode_per_batch == 0 and i == 0:
                    batch_loss = loss
                elif i % args.episode_per_batch == 0 and i != 0:
                    batch_loss.backward()
                    batch_loss = loss
                else:
                    batch_loss += loss
            elif args.grad_setting == 'batch_mean_grad':
                loss = loss / args.episode_per_batch
            optimizer.step()

            # calculate accuracy
            acc = (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            train_acc.append(acc)
            total_acc = sum(train_acc) / len(train_acc)

            # print result
            printer("train", e, args.num_epochs, i+1, len(train_loader), loss.item(), total_loss, acc * 100, total_acc * 100)

            # tensorboard
            writer.add_scalar("Loss/train", loss.item(), n_iter_train)
            writer.add_scalar("Accuracy/train", acc, n_iter_train)
            n_iter_train += 1
            #break
        print("")
        val_acc = []
        val_loss = []
        model.eval()
        with torch.no_grad():
            for i, (datas, modal_aux, _) in enumerate(val_loader):
                aux = dict()
                for modal in ['depth', 'pose', 'flow']:
                    if modal in modal_aux:
                        aux[modal] = modal_aux[modal].to(device)

                datas = datas.to(device)
                pivot = args.way * args.shot
  
                shot, query = datas[:pivot], datas[pivot:]
                labels = torch.arange(args.way).repeat(args.query).to(device)
                support_labels = torch.arange(args.way).repeat(args.query).to(device)     

                # calculate loss
                # onehot_labels = Variable(torch.zeros(args.way*args.query, args.way).scatter_(1, torch.arange(args.way).repeat(args.query).view(-1, 1), 1)).to(device)
                if args.model == 'proto':
                    pred = model(shot, query, aux)
                    loss = F.cross_entropy(pred, labels).item()
                    # loss = F.mse_loss(pred, onehot_labels).item()
                elif args.model == 'trx':
                    pred = model(shot, query, support_labels, aux)
                    loss = cross_entropy_trx(pred, labels, device)

                val_loss.append(loss)
                total_loss = sum(val_loss)/len(val_loss)

                # calculate accuracy
                acc = (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
                val_acc.append(acc)
                total_acc = sum(val_acc)/len(val_acc)

                # print result
                printer("val", e, args.num_epochs, i+1, len(val_loader), loss, total_loss, acc * 100, total_acc * 100)

                # tensorboard
                writer.add_scalar("Loss/val", loss, n_iter_val)
                writer.add_scalar("Accuracy/val", acc, n_iter_val)
                n_iter_val += 1

            if total_acc > best:
                best = total_acc
                torch.save(model.state_dict(), os.path.join(args.save_path, "best.pth"))
            torch.save(model.state_dict(), os.path.join(args.save_path, "last.pth"))
            print("Best: {:.2f}%".format(best * 100))

        lr_scheduler.step()