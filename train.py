import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from meta_trainer import meta_trainer
from dataset.prepare_dataset import prepare_dataset

# import networks
from nets.supcon import SupConResNet
from nets.cycle_net import CycleNet
from nets.meta_nets import MetaProtos
from nets.trx import TRX

from utils.loss import SupConLoss, PatchCycleLoss, CE_TRX
from utils.visualize import vis_salient_patch
from utils.utils import precedding, printer
from utils.configs import init_parser, add_args


max_acc = 0
n_iter_train = 0
n_iter_val = 0 
if __name__ == "__main__":
    parser = init_parser()
    # meta-learning: Proto, Relation, MatchNet
    # previous FSL-action recognition: TRX
    # fine-tuning: cycle, supcon
    parser.add_argument("--method", type=str, default="proto")  
    
    # ===========================Few-shot options=================================
    parser.add_argument("--meta_learn", action="store_true")
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--num_train_episode", type=int, default=5000)
    parser.add_argument("--num_val_episode", type=int, default=1000)
    parser.add_argument("--episode_per_batch", type=int, default=16)
    parser.add_argument("--metric", type=str, default="cosine")  # euclidean, relation, cosine
    parser.add_argument("--classifier", type=str, default='LR')  # LR, NN, Cosine, Proto, SVM
    # ===========================Key training options==============================
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.0025)
    parser.add_argument("--scheduler_epoch_size", type=int, default=5)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--bn_threshold", type=float, default=2e-2)   
    # ===========================Contrastive options========================
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--contrast_loss", type=str, default='SupCon')
    parser.add_argument("--use_ce", action="store_true")
    parser.add_argument("--use_contrast", action="store_true")
    parser.add_argument("--sigma_ce", type=float, default=1.0)
    parser.add_argument("--sigma_contrast", type=float, default=1.0)
    # ===========================Cycle consistency options========================
    parser.add_argument("--epoch_iter", type=int, default=120000)
    parser.add_argument("--save_iter", type=int, default=500)
    parser.add_argument("--num_classes", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--sim_thresh", type=float, default=0.8)
    parser.add_argument("--iter_reset", type=int, default=10000)
    parser.add_argument("--sigma_sp", type=float, default=0.1)
    parser.add_argument("--sigma_feat", type=float, default=0.2)
    # ===========================Multi-modal options==============================
    parser.add_argument("--multi_modal", type=bool, default=False)
    parser.add_argument("--use_depth", type=bool, default=False)
    parser.add_argument("--use_pose", type=bool, default=False)
    parser.add_argument("--use_flow", type=bool, default=False)
    parser.add_argument("--fusion", type=str, default=None)  # None, "cen"
    parser.add_argument("--sharing", action="store_true")
     
    args = add_args(parser)

    precedding(args)
    writer = SummaryWriter(args.tensorboard_path)

    # ==========
    # Dataset
    # ==========
    train_loader, val_loader, id2clss = prepare_dataset(args)

    # ============
    # Model Build
    # ============
    # select a model, i prepaired two models for simple testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.method == 'supcon':
        model = SupConResNet(args, num_gpus=args.num_gpus)
        criterion = SupConLoss(args)
    elif args.method == 'proto':
        model = MetaProtos(args)
        criterion = nn.CrossEntropyLoss()
        #criterion = F.cross_entropy
    elif args.method == 'trx':
        model = TRX(args, way=args.way, shot=args.shot, query=args.query)
        criterion = CE_TRX
    elif args.method == 'cycle':
        model = CycleNet(args, device)
        criterion = PatchCycleLoss(args)
    model = model.to(device)
    if args.num_gpus > 1:
        model.distribute_model(args.num_gpus)
    
    # =============
    # Optimization
    # =============
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_epoch_size, gamma=args.scheduler_gamma)
    
    # ===============
    # Train and Eval
    # ===============
    for e in range(args.num_epochs):
        tips = [e, n_iter_train, id2clss]
        n_iter_train = meta_trainer(args, model, criterion, optimizer, train_loader, device, writer, tips)
        
        torch.save(model.state_dict(), os.path.join(args.save_path, "%d.pth"%e))
        lr_scheduler.step()

        if (e+1) % 2 == 0:
            meta_eval(e, args, val_loader, model, max_acc)
            
        
def meta_eval(e, args, val_loader, model, max_acc, n_iter_val):
    print("\nTest Epoch_{}... {}-way {}-shot {}-query".format(e, args.way, args.shot, args.query))
    val_acc = []
    model.eval()
    with torch.no_grad():
        for i, (_datas, modal_aux) in enumerate(val_loader):
            _labels = torch.arange(args.way).repeat(args.shot+args.query).to(device)
            query_labels = _labels[args.way*args.shot:]

            aux = dict()
            for modal in ['depth', 'pose', 'flow']:
                if modal in modal_aux:
                    aux[modal] = modal_aux[modal].to(device)
            _datas = _datas.to(device)  # way*shot+way*query

            if args.method != 'trx':
                pred = model(_datas, aux, _labels, is_eval=True)
            else:
                logits, pred = model(_datas, aux, _labels[args.way*args.shot:])

            acc = (pred.argmax(1) == query_labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()

            val_acc.append(acc)
            total_acc = sum(val_acc) / len(val_acc)

            printer("val", e, args.num_epochs, i+1, len(val_loader), 0, 0, acc * 100, total_acc * 100, max_acc=max_acc)
        if total_acc > max_acc:
            torch.save(model.state_dict(), os.path.join(args.save_path, "best.pth"))
            max_acc = total_acc
        writer.add_scalar("Accuracy/val", acc, n_iter_val)
        print("\n")
