import argparse


def init_parser():
    parser = argparse.ArgumentParser()
    # ===========================Basic options===================================
    parser.add_argument("--dataset", type=str, default="kinetics100")
    parser.add_argument("--class_split_folder", type=str, default="data/splits/kinetics100/")
    parser.add_argument("--frames_path", type=str, default="data/images/kinetics100/")
    parser.add_argument("--labels_path", type=str, default="data/splits/kinetics100/")
    parser.add_argument("--save_path", type=str, default="ckpt/train1/")
    parser.add_argument("--tensorboard_path", type=str, default="./data/tensorboard/train1")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--visualize", action="store_true")
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

    return parser


def add_args(parser):
    # ===========================TRX options=================================
    parser.add_argument('--temp_set', nargs='+', type=int, help='cardinalities e.g. 2,3 is pairs and triples', default=[2,3])
    parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
    parser.add_argument("--trans_linear_out_dim", type=int, default=1152, help="Transformer linear_out_dim")

    args = parser.parse_args()

    if args.backbone == "resnet50" or args.backbone == "resnet34":
        args.img_size = 224
    if args.backbone == "resnet50":
        args.trans_linear_in_dim = 2048
    else:
        args.trans_linear_in_dim = 512

    return args