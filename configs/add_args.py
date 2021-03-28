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