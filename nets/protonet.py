import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .backbones.resnet import resnet18, resnet34, resnet50
#from .backbones.resnet_basic import resnet18
from torchvision.models.video import r2plus1d_18
from torchvision.models.video.resnet import BasicBlock, Conv2Plus1D
from .utils import freeze_all, freeze_layer, freeze_bn, initialize_linear, initialize_3d


class ProtoNet(nn.Module):
    def __init__(self, args, way=5, shot=1, query=5, hidden_size=512, num_layers=1, bidirectional=True):
        super(ProtoNet, self).__init__()
        self.way = way
        self.shot = shot
        self.query = query
        self.metric = args.metric
        
        # multi-modal option
        self.multi_modal = args.multi_modal
        self.use_depth = args.use_depth
        self.use_pose = args.use_pose
        self.use_flow = args.use_flow
        self.fusion = args.fusion
        self.sharing = args.sharing
        
        if not self.sharing:
            self.use_depth = False
            self.use_pose = False
            self.use_flow = False
            self.fusion = None

        self.num_modal = 1 if not self.multi_modal else 1 + int(self.use_depth)+int(self.use_pose)+int(self.use_flow)

        # resnet18(freezing)
        self.backbone = args.backbone
        self.freeze_all = args.freeze_all
        self.resnet = self.build_backbone(args.bn_threshold)
        if not self.sharing:
            self.resnet2 = self.build_backbone(args.bn_threshold)

        # gru
        self.hidden_size = hidden_size * self.num_modal
        self.gru = nn.GRU(input_size=self.last_dim*self.num_modal, hidden_size=hidden_size, batch_first=True, 
                          num_layers=num_layers, dropout=0.5 if num_layers > 1 else 0, bidirectional=bidirectional)

        # linear
        self.linear = nn.Linear(int(hidden_size*2) if bidirectional else hidden_size, hidden_size)
        self.linear.apply(initialize_linear)

        # relation module
        
        # scaler
        self.scaler = nn.Parameter(torch.tensor(5.0))

    def forward(self, shot, query, aux):
        x = torch.cat((shot, query), dim=0)
        b, t, c, h, w = x.shape  # way*(shot+query), num_frames, channel, height, weight
        x = x.view(b * t, c, h, w)

        if self.multi_modal:
            mm_x= [x]
            if self.use_depth:
                depth = aux['depth'].view(b * t, c, h ,w)
                mm_x = mm_x + [depth]
            if self.use_pose:
                pose = aux['pose'].view(b * t, c, h ,w)
                mm_x = mm_x + [pose]
            if self.use_flow:
                flow = aux['flow'].view(b * t, c, h ,w)
                mm_x = mm_x + [flow]
            x = mm_x

        # encoder
        if self.sharing:
            x = self.resnet(x)
        else:
            x = self.resnet([x[0]]) + self.resnet2([x[1]])
        x = torch.cat(x, dim=1).squeeze() 
    
        # gru
        x = x.view(b, t, self.last_dim*self.num_modal)
        x = (self.gru(x)[0]).mean(1) # this may be helful for generalization
    
        # linear
        x = self.linear(x)

        shot, query = x[:shot.size(0)], x[shot.size(0):]
        shot = shot.reshape(self.shot, self.way, -1).mean(dim=0)
        
        # make prototype
        # matching stage
        if self.metric == 'cosine':
            shot = F.normalize(shot, dim=-1)
            query = F.normalize(query, dim=-1)        
            logits = torch.mm(query, shot.t())
        elif self.metric == 'euclidean':
            shot = shot.unsqueeze(0).repeat(self.way*self.query, 1, 1)
            query = query.unsqueeze(1).repeat(1, self.way, 1)
            logits = -((shot - query)**2).sum(dim=-1)
        
        return logits# * self.scaler
    
    def build_backbone(self, bn_threshold):
        if self.multi_modal:
            num_parallel = 1+int(self.use_depth)+int(self.use_pose)+int(self.use_flow)

            if self.backbone == "resnet18":
                resnet = resnet18(pretrained=True, parallel=num_parallel, num_parallel=num_parallel, 
                                  bn_threshold=bn_threshold, fusion=self.fusion) 
            elif self.backbone == "resnet34":
                resnet = resnet34(pretrained=True, parallel=num_parallel, num_parallel=num_parallel, 
                                  bn_threshold=bn_threshold, fusion=self.fusion)
            elif self.backbone == "resnet50":
                resnet = resnet50(pretrained=True, parallel=num_parallel, num_parallel=num_parallel, 
                                  bn_threshold=bn_threshold, fusion=self.fusion)
        else:
            if self.backbone == "resnet18":
                resnet = models.resnet18(pretrained=True)  
            elif self.backbone == "resnet34":
                resnet = models.resnet34(pretrained=True)
            elif self.backbone == "resnet50":
                resnet = models.resnet50(pretrained=True)
        last_layer_idx = -1

        
        #self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        '''
        self.resnet = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        '''
            
        self.last_dim = resnet.fc.in_features

        if self.freeze_all:
            resnet.apply(freeze_all)

        return resnet

    def _downsample(self, inplanes, outplanes):
        return nn.Sequential(
            nn.Conv3d(inplanes, outplanes, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(outplanes),
        )

    def distribute_model(self, num_gpus):
        """
        Distribte the backbone over multiple GPUs
        """
        if num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(num_gpus)])