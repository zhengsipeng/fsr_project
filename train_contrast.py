import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nets.resnet_big import SupConResNet
from utils.utils import path_check, args_print_save, printer, printer_cycle, AverageMeter
from utils.visualize import vis_salient_patch
from utils.loss import SupConLoss
from dataset.contrast_reader import ContrastDataset
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
    parser.add_argument("--episode_per_batch", type=int, default=16)
    parser.add_argument("--grad_setting", type=str, default='basic')
    parser.add_argument("--visualize", action="store_true")
    # ===========================Few-shot options=================================
    parser.add_argument("--model", type=str, default="proto")  # proto, trx
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--num_train_episode", type=int, default=7000)
    parser.add_argument("--num_val_episode", type=int, default=1000)
    parser.add_argument("--metric", type=str, default="cosine")  # euclidean, relation, cosine
    # ===========================Contrastive options========================
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--contrast_loss", type=str, default='SupCon')
    parser.add_argument("--use_ce", action="store_true")
    parser.add_argument("--use_contrast", action="store_true")
    parser.add_argument("--sigma_contrast", type=float, default=1)
    parser.add_argument("--sigma_ce", type=float, default=1)
    # ===========================Cycle consistency options========================
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--epoch_iter", type=int, default=120000)
    parser.add_argument("--save_iter", type=int, default=500)
    parser.add_argument("--num_classes", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--is_pretrain", type=bool, default=True)
    parser.add_argument("--classifier", type=str, default='LR')
    # ===========================Multi-modal options==============================
    parser.add_argument("--multi_modal", type=bool, default=False)
    parser.add_argument("--use_depth", type=bool, default=False)
    parser.add_argument("--use_pose", type=bool, default=False)
    parser.add_argument("--use_flow", type=bool, default=False)
    parser.add_argument("--fusion", type=str, default=None)  # None, "cen"
    parser.add_argument("--sharing", action="store_true")
    # ===========================Dataset options==================================
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--random_pad_sample", action="store_true")
    parser.add_argument("--pad_option", type=str, default="default")
    parser.add_argument("--uniform_frame_sample", action="store_true")
    parser.add_argument("--random_start_position", action="store_true")
    parser.add_argument("--max_interval", type=int, default=64)
    parser.add_argument("--random_interval", action="store_true")
    # ===========================Backbone options=================================
    parser.add_argument("--backbone", type=str, default="resnet50")  # resnet18, resnet34, resnet50
    parser.add_argument("--freeze_all", type=bool, default=False)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.0025)
    parser.add_argument("--scheduler_epoch_size", type=int, default=5)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--bn_threshold", type=float, default=2e-2)    
    args = add_args(parser)

    # check options
    assert args.model in ["proto", "trx"], "'{}' model is invalid.".format(args.model)
    assert args.backbone in ["resnet18", "resnet34", "resnet50", "r2plus1d"], "'{}' backbone is invalid.".format(args.model)
    assert args.metric in ["cosine", "euclidean", "relation"], "'{}' metric is invalid.".format(args.metric)

    # path to save
    path_check(args.save_path)
    
    # path to tensorboard
    writer = SummaryWriter(args.tensorboard_path)
    
    # print args and save it in the save_path
    args_print_save(args)

    # ==========
    # Dataset
    # ==========
    train_dataset = ContrastDataset(
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

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, sampler=None)     
    val_sampler = EpisodeSampler(val_dataset.classes, args.num_val_episode, args.way, args.shot, args.query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

    # ============
    # Model Build
    # ============
    # select a model, i prepaired two models for simple testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupConResNet(args, num_gpus=args.num_gpus)
    model = model.to(device)
    if args.num_gpus > 1:
        model.distribute_model(args.num_gpus)
    
    # =============
    # Optimization
    # =============
    criterion = SupConLoss(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_epoch_size, gamma=args.scheduler_gamma)
    
    losses = AverageMeter()
    max_acc = 0
    for e in range(args.num_epochs):
        train_loss = []
        train_acc = []
        model.train()
        for idx, (datas, labels) in enumerate(train_loader):
            if (idx+1) % 150 == 0:
                # we evaluate the network with meta-learning set using meta-val
                print("\nTest... {}-way {}-shot {}-query".format(args.way, args.shot, args.query))
                val_acc = []
                val_loss = []
                model.eval()
                with torch.no_grad():
                    for i, (_datas, modal_aux) in enumerate(val_loader):
                        if i > 100:
                            break
                        _labels = torch.arange(args.way).repeat(args.shot+args.query).to(device)

                        aux = dict()
                        for modal in ['depth', 'pose', 'flow']:
                            if modal in modal_aux:
                                aux[modal] = modal_aux[modal].to(device)
                        _datas = _datas.to(device)  # way*shot+way*query

                        acc = model(_datas, aux, _labels, is_pretrain=False)
                        val_acc.append(acc)
                        total_acc = sum(val_acc) / len(val_acc)

                        printer("val", e, args.num_epochs, i+1, len(val_loader), 0, 0, acc * 100, total_acc * 100)
                    print("\n")

            # datas: 2, bsz, t, 3, h, w
            datas = torch.cat([datas[0], datas[1]], dim=0)
            datas = datas.to(device)  # batchsize*2, T, C, H, W
            #assert 1==0
            #print(datas.shape)
            labels = labels.to(device)
            bsz = labels.shape[0]

            logits, features = model(datas)
            acc = (logits.argmax(1)==labels).mean().item()
            train_acc.append(acc)
            total_acc = sum(train_acc) / len(train_acc)
            # SupCon, SimCLR
            loss, ce_loss, contrast_loss = criterion(features, logits, labels, loss_type=args.contrast_loss)
            losses.update(loss.item(), bsz)

            # update weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.use_ce:
                print('Train: [epoch: %d][iter: %d][loss: %.4f][ce: %.3f][contrast: %.3f][avg_loss: %.3f][acc: %.3f]'%(e, idx, losses.val, ce_loss.item(), contrast_loss.item(), losses.avg, total_acc*100))
            else:
                print('Train: [epoch: %d][iter: %d][loss: %.4f][avg_loss: %.4f]'%(e, idx, losses.val, losses.avg))

            torch.cuda.empty_cache()

        torch.save(model.state_dict(), os.path.join(args.save_path, "%d.pth"%e))

        lr_scheduler.step()
        if (e+1) % 2 == 0:
            # we evaluate the network with meta-learning set using meta-val
            print("\nTest Epoch_{}... {}-way {}-shot {}-query".format(e, args.way, args.shot, args.query))
            val_acc = []
            val_loss = []
            model.eval()
            with torch.no_grad():
                for i, (_datas, modal_aux) in enumerate(val_loader):
                    _labels = torch.arange(args.way).repeat(args.shot+args.query).to(device)

                    aux = dict()
                    for modal in ['depth', 'pose', 'flow']:
                        if modal in modal_aux:
                            aux[modal] = modal_aux[modal].to(device)
                    _datas = _datas.to(device)  # way*shot+way*query

                    acc = model(_datas, aux, _labels, is_pretrain=False)
                    val_acc.append(acc)
                    total_acc = sum(val_acc) / len(val_acc)

                    printer("val", e, args.num_epochs, i+1, len(val_loader), 0, 0, acc * 100, total_acc * 100, max_acc=max_acc)
                if total_acc > max_acc:
                    max_acc = total_acc
                print("\n")

        
                