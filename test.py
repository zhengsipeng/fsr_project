import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from nets.protonet import ProtoNet
from nets.trx import trx
from dataset.general_reader import GeneralDataset, EpisodeSampler
from configs.add_args import add_args
from utils.utils import printer, mean_confidence_interval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ===========================Basic options===================================
    parser.add_argument("--dataset", type=str, default="kinetics100")
    parser.add_argument("--frames_path", type=str, default="data/images/ucf101")
    parser.add_argument("--labels_path", type=str, default="data/splits/ucf101")
    parser.add_argument("--load_path", type=str, default="./save/train1")
    parser.add_argument("--use_best", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=4)
    # ===========================Dataset options==================================
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--sequence_length", type=int, default=35)
    #parser.add_argument("--num-epochs", type=int, default=10)
    # ===========================Few-shot options=================================
    parser.add_argument("--model", type=str, default='trx')
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--num_test_episode", type=int, default=10000)
    parser.add_argument("--metric", type=str, default="cosine")  # euclidean, relation, cosine
    # ===========================Multi-modal options==============================
    parser.add_argument("--multi_modal", type=bool, default=True)
    parser.add_argument("--use_depth", type=bool, default=False)
    parser.add_argument("--use_pose", type=bool, default=False)
    parser.add_argument("--use_flow", type=bool, default=False)
    # ===========================Backbone options=================================
    parser.add_argument("--backbone", type=str, default="resnet18")  # resnet18, resnet34, resnet50
    parser.add_argument("--freeze_all", type=bool, default=False)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scheduler_step_size", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.9)
    parser.add_argument("--fusion", type=str, default=None)
    parser.add_argument("--bn_threshold", type=float, default=2e-2)    

    args = parser.parse_args()

    # check options
    assert args.model in ["proto", "trx"], "'{}' model is invalid.".format(args.model)
    assert args.backbone in ["resnet18", "resnet34", "resnet50", "r2plus1d"], "'{}' backbone is invalid.".format(args.model)
    assert args.metric in ["cosine", "euclidean", "relation"], "'{}' metric is invalid.".format(args.metric)

    # =========
    # Dataset
    # =========
    test_dataset = GeneralDataset(
        args,
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        setname='test',
        # pad option
        random_pad_sample=False,
        pad_option='default',
        # frame sampler option
        uniform_frame_sample=True,
        random_start_position=False,
        max_interval=7,
        random_interval=False,
    )

    print("[test] number of videos / classes: {} / {}".format(len(test_dataset), test_dataset.num_classes))
    print("total testing episodes: {}".format(args.num_test_episode))
    test_sampler = EpisodeSampler(test_dataset.classes, args.num_test_episode, args.way, args.shot, args.query)
    test_loader = DataLoader(dataset=test_dataset, batch_sampler=test_sampler, num_workers=0 if os.name == 'nt' else 4, pin_memory=True)

    # =============
    # Model Build
    # ============= 
    if args.use_best:
        load_path = os.path.join(args.load_path, "best.pth")
    else:
        load_path = os.path.join(args.load_path, "last.pth")
    # load_path check
    assert os.path.exists(load_path), "'{}' file is not exists !!".format(load_path)

    if args.model == 'proto':
        model = ProtoNet(
            args
            way=args.way,
            shot=args.shot,
            query=args.query,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            bidirectional=args.bidirectional,
        )
    elif args.model == 'trx':
        model = TRX(
            args,
            way=args.way,
            shot=args.shot,
            query=args.query,
        )
    else:
        raise NotImplementedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if args.num_gous > 1:
        model.distribute_model(args.num_gpus)
    
    model.eval()
    total_loss = 0
    epoch_acc = 0
    total_acc = []
    print("test... {}-way {}-shot {}-query".format(args.way, args.shot, args.query))
    with torch.no_grad():
        test_acc = []
        test_loss = []
        for i, (datas, modal_aux, _) in enumerate(test_loader):
            # prepare multi-modal auxilary
            aux = dict()
            for modal in ['depth', 'pose', 'flow']:
                if modal in modal_aux:
                    aux[modal] = modal_aux[modal].to(device)

            datas = datas.to(device)
            pivot = args.way * args.shot
            shot, query = datas[:pivot], datas[pivot:]
            support_labels = torch.arange(args.way).repeat(args.shot).to(device)
            labels = torch.arange(args.way).repeat(args.query).to(device)
            # one_hot_labels = Variable(torch.zeros(args.way*args.query, args.way).scatter_(1, labels.view(-1, 1), 1)).to(device)
            if args.model == 'proto':
                pred = model(shot, query, aux)
                loss = F.cross_entropy(pred, labels)
            elif args.model == 'trx':
                pred = model(shot, query, support_labels, aux)
                loss = cross_entropy_trx(pred, labels, device)

            # calculate loss
            test_loss.append(loss.item())
            total_loss = sum(test_loss)/len(test_loss)

            # calculate accuracy
            acc = 100 * (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            test_acc.append(acc)
            total_acc.append(acc)
            avg_acc = sum(test_acc)/len(test_acc)
            printer("test", e, i+1, len(test_loader), loss, total_loss, acc, avg_acc)

        # get mean confidence interval per epochs
        m, h = mean_confidence_interval(test_acc, confidence=0.95)
        print(" => {} episodes [{:.2f} +-{:.2f}]".format(args.num_test_episode, m, h))

        # get total mean confidence interval
        m, h = mean_confidence_interval(total_acc, confidence=0.95)
        print("total {} episodes Result: {:.2f}+-{:.2f}".format(args.num_test_episode, m, h))