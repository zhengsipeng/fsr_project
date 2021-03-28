import torch
import os
import sys
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nets.cyclenet import CycleNet
from utils.utils import path_check, args_print_save, printer, printer_cycle
from utils.loss import PatchCycleLoss
from dataset.cycle_reader import CycleDataset
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
    # ===========================Few-shot options=================================
    parser.add_argument("--model", type=str, default="proto")  # proto, trx
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--num_train_episode", type=int, default=7000)
    parser.add_argument("--num_val_episode", type=int, default=3000)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--metric", type=str, default="cosine")  # euclidean, relation, cosine
    # ===========================Cycle consistency options========================
    parser.add_argument("--epoch_iter", type=int, default=640000)
    parser.add_argument("--save_iter", type=int, default=6000)
    parser.add_argument("--num_classes", type=int, default=64)
    parser.add_argument("--sim_thresh", type=float, default=0.8)
    parser.add_argument("--iter_reset", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--is_pretrain", type=bool, default=True)
    parser.add_argument("--classifier", type=str, default='LR')
    parser.add_argument("--sigma_ce", type=float, default=1)
    parser.add_argument("--sigma_sp", type=float, default=0.1)
    parser.add_argument("--sigma_feat", type=float, default=0.4)
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
    # pad options
    parser.add_argument("--random_pad_sample", action="store_true")
    parser.add_argument("--pad_option", type=str, default="default")
    # frame options
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
    parser.add_argument("--scheduler_epoch_size", type=int, default=1)
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
    
    # dataset for validation follows the meta-learning setting 
    # do not use the autoaugment on the validation or test dataset
    val_dataset = GeneralDataset(
        args,
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        setname='val',
        random_pad_sample=False,
        pad_option='default',
        uniform_frame_sample=True,
        random_start_position=False,
        max_interval=7,
        random_interval=False,
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)     
    val_sampler = EpisodeSampler(val_dataset.classes, args.num_val_episode, args.way, args.shot, args.query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=0 if os.name == 'nt' else 4, pin_memory=True)

    # ============
    # Model Build
    # ============
    # select a model, i prepaired two models for simple testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CycleNet(args, device)
    model = model.to(device)
    if args.num_gpus > 1:
        model.distribute_model(args.num_gpus)
    
    # =============
    # Optimization
    # =============
    criterion = PatchCycleLoss(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_epoch_size, gamma=args.scheduler_gamma)

    best = 0 # top1 best accuracy
    total_loss, total_ce, total_infonce_sp, total_infonce = 0, 0, 0, 0
    total_acc = 0
    n_iter_train = 0
    n_iter_val = 0
    avg_pos_num = 0
    for e in range(1, args.num_epochs+1):
        train_loss, train_ce, train_infonce_sp, train_infonce = [], [], [], []
        train_acc = []
        model.train()
        for i, (datas, modal_aux, labels) in enumerate(train_loader):
            # check the parameters update
            #params = list(model.named_parameters())#get the index by debuging
            #print(params[0][1].data)#data
            #print(params[-4][0])

            if i > args.epoch_iter / args.batch_size:
                break

            # prepare multi-modal auxilary
            aux = dict()
            for modal in ['depth', 'pose', 'flow']:
                if modal in modal_aux:
                    aux[modal] = modal_aux[modal].to(device)

            datas = datas.to(device)  # batchsize, 2, C, H, W
            labels = labels.to(device)
            
            logits, st_locs, st_locs_back, log_p_sim, log_p_back_sim, pos_onehot = model(datas, aux)
            pos_num = pos_onehot.sum(1).float().mean()
            #if pos_num > 100:
            #    args.sim_thresh += args.sim_thresh * 0.1
            #print(pos_onehot.sum(1))
            loss, ce_loss, infonce_sp_loss, info_nce_loss = criterion(logits, \
                        labels, st_locs, st_locs_back, log_p_sim, log_p_back_sim, pos_onehot)
            
            # update weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate the loss and accuracy
            train_loss.append(loss.item())
            train_ce.append(ce_loss.item())
            train_infonce_sp.append(infonce_sp_loss.item())
            train_infonce.append(info_nce_loss.item())
            total_loss = sum(train_loss)/len(train_loss)
            total_ce = sum(train_ce)/len(train_loss)
            total_infonce_sp = sum(train_infonce_sp)/len(train_loss)
            total_infonce = sum(train_infonce)/len(train_loss)

            acc = (logits.argmax(1)==labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            train_acc.append(acc)
            total_acc = sum(train_acc) / len(train_acc)

            printer_cycle("train", e, args.num_epochs, i+1, len(train_loader), 
                           loss.item(), total_loss, ce_loss.item(), total_ce,
                           infonce_sp_loss.item(), total_infonce_sp, info_nce_loss.item(), total_infonce,
                           acc*100, total_acc*100, pos_num, args.sim_thresh)
            
            # tensorboard
            writer.add_scalar("Loss/train", loss.item(), n_iter_train)
            writer.add_scalar("Accuracy/train", acc, n_iter_train)
            n_iter_train += 1

            # save the checkpoint
            if i+1 % args.save_iter == 0:
                torch.save(model.state_dict(), os.path.join(args.save_path, "%d_%d.pth"%(e, args.epoch_iter)))
            #if i > 10:
            #break

        torch.save(model.state_dict(), os.path.join(args.save_path, "%d.pth"%e))

        lr_scheduler.step()
        #print(optimizer.state_dict()['param_groups'][0]['lr'])

        # we evaluate the network with meta-learning set using meta-val
        print("\nVal... {}-way {}-shot {}-query".format(args.way, args.shot, args.query))
        val_acc = []
        val_loss = []
        model.eval()
        with torch.no_grad():
            for i, (datas, modal_aux) in enumerate(val_loader):
                labels = torch.arange(args.way).repeat(args.shot+args.query).to(device)

                aux = dict()
                for modal in ['depth', 'pose', 'flow']:
                    if modal in modal_aux:
                        aux[modal] = modal_aux[modal].to(device)
                datas = datas.to(device)  # way*shot+way*query

                # do validation
                acc = model(datas, aux, labels, is_pretrain=False)
                val_acc.append(acc)
                total_acc = sum(val_acc) / len(val_acc)

                printer("val", e, args.num_epochs, i+1, len(val_loader), 0, 0, acc * 100, total_acc * 100)
            
                writer.add_scalar("Accuracy/val", acc, n_iter_val)
                n_iter_val += 1
                #if i > 10:
                #break