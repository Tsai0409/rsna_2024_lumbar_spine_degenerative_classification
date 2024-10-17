import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F

from .backbones import *
from .senet import *
from .activation import *
from .layers import *
from .self_attention import SelfAttention

def drop_fc(model):
    if model.__class__.__name__ in ['Xcit', 'Cait']:
        nc = model.head.in_features
        model.norm = nn.Identity()
        model.head_drop = nn.Identity()
        model.head = nn.Identity()
        new_model = model
    elif model.__class__.__name__ == 'RegNet':
        nc = model.head.fc.in_features
        model.head.global_pool = nn.Identity()
        model.head.flatten = nn.Identity()
        model.head.drop = nn.Identity()
        model.head.fc = nn.Identity()
        new_model = model
    elif model.__class__.__name__ == 'MetaFormer':
        nc = model.head.fc.fc1.in_features
        model.head.global_pool = nn.Identity()
        model.head.norm = nn.Identity()
        model.head.flatten = nn.Identity()
        model.head.drop = nn.Identity()
        model.head.fc = nn.Identity()
        new_model = model
    elif model.__class__.__name__ == 'Sequencer2D':
        nc = model.head.in_features
        model.head = nn.Identity()
        new_model = model
    elif model.__class__.__name__ == 'Sam':
        new_model = model.image_encoder
        nc = new_model.neck[0].out_channels
    elif 'SwinTransformer' in model.__class__.__name__:
        nc = model.head.in_features
        model.head = nn.Identity()
        new_model = model
    elif model.__class__.__name__ == 'ConvNeXt':
        nc = model.head.fc.in_features
        model.head.global_pool = nn.Identity()
        model.head.norm = nn.Identity()
        model.head.flatten = nn.Identity()
        model.head.drop = nn.Identity()
        model.head.fc = nn.Identity()
        new_model = model
    elif model.__class__.__name__ == 'FeatureEfficientNet':
        new_model = model
        nc = model._fc.in_features
    elif model.__class__.__name__ == 'RegNetX':
        new_model = nn.Sequential(*list(model.children())[0])[:-1]
        nc = list(model.children())[0][-1].fc.in_features
    elif model.__class__.__name__ == 'DenseNet':
        new_model = nn.Sequential(*list(model.children())[:-1])
        nc = list(model.children())[-1].in_features
    # elif model.__class__.__name__ == 'EfficientNet':
    #     new_model = nn.Sequential(*list(model.children())[:-2])
    #     import pdb;pdb.set_trace()
    #     nc = 1280
    else:
        new_model = nn.Sequential(*list(model.children())[:-2])
        nc = list(model.children())[-1].in_features
    return new_model, nc

class MultiInstanceCNNModelRetrain(nn.Module):
    def __init__(
        self,
        num_classes=1,
        n_instance=5,
        center_loss_feat_dim=512,
        base_model=senet_mod(se_resnext50_32x4d, pretrained=True),
    ):
        super().__init__()
        self.n_instance = n_instance
        self.criterion = nn.CrossEntropyLoss()
        self.center_loss_feat_dim = center_loss_feat_dim
        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.Linear(2 * nc, self.center_loss_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(self.center_loss_feat_dim, num_classes)

    def forward(self, x):
        # st()
        # np.all(np.load('/groups/gca50041/ariyasu/rsna/results/c.npy') == x.cpu().numpy())
        # np.save('/groups/gca50041/ariyasu/rsna/results/c.npy', x.cpu().numpy())
        bs, n, ch, w, h = x.shape # 8, 5, 3, 384, 384
        x = x.view(bs * n, ch, w, h) # 40, 3, 384, 384
        x = self.encoder(x) # 40, 2048, 20, 20
        bs2, ch2, w2, h2 = x.shape

        x = (
            x.view(-1, n, ch2, w2, h2)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(bs, ch2, n * w2, h2)
        )  # x: bs x C' x N W'' x W''
        x = self.head(x)
        x = self.fc(x)
        return x

class MultiInstanceCNNModelRetrainForSwin(nn.Module):
    def __init__(
        self,
        num_classes=1,
        n_instance=5,
        center_loss_feat_dim=512,
        base_model=senet_mod(se_resnext50_32x4d, pretrained=True),
        use_max_pool=True
    ):
        super().__init__()
        self.n_instance = n_instance
        self.criterion = nn.CrossEntropyLoss()
        self.center_loss_feat_dim = center_loss_feat_dim
        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        self.use_max_pool = use_max_pool
        if use_max_pool:
            self.fc = nn.Linear(nc*2, num_classes)
        else:
            self.fc = nn.Linear(nc, num_classes)

    def forward(self, x):
        bs, n, ch, w, h = x.shape # 4,5,3,384,384
        x = x.view(bs * n, ch, w, h)  # 20,3,384,384
        x = self.encoder(x)  # 20,12,12,1024
        bsn, w2, h2, ch2 = x.size()
        x = x.view(bs, n*w2, h2, ch2)
        mean_ = x.mean(1).mean(1)
        if self.use_max_pool:
            max_ = x.max(1)[0].max(1)[0]
            x = torch.cat([mean_, max_], 1)
        else:
            x = mean_
        x = self.fc(x)
        return x

'''
Models
'''

class MultiInstanceModel(nn.Module):
    def __init__(self, base_model, num_classes=2, in_channels=3, effnet=False):
        super(MultiInstanceModel, self).__init__()

        self.model_name = base_model.__class__.__name__
        self.effnet = effnet
        self.encoder, nc = drop_fc(base_model)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            AdaptiveConcatPool2d(), Flatten(),
            nn.Linear(2*nc, 512), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(512, num_classes)
        if in_channels != 3:
            self.encoder[0].conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    def forward(self, x):
        # x: bs x N x C x W x W
        bs, n, ch, w, h = x.shape
        x = x.view(bs*n, ch, w, h) # x: N bs x C x W x W
        x = self.encoder(x) # x: N bs x C' x W' x W'

        # Concat and pool
        bs2, ch2, w2, h2 = x.shape
        x = x.view(-1, n, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
            .contiguous().view(bs, ch2, n*w2, h2) # x: bs x C' x N W'' x W''
        feature_output = self.head(x)

        x = self.fc(feature_output)
        return feature_output, x

    def __repr__(self):
        return f'MIL({self.model_name})'

class MultiInstanceModelWithWataruAttention(nn.Module):
    def __init__(self, base_model, num_classes=2, in_channels=3, effnet=False):
        super(MultiInstanceModelWithWataruAttention, self).__init__()

        self.model_name = base_model.__class__.__name__
        self.effnet = effnet
        self.encoder, nc = drop_fc(base_model)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.self_attention = SelfAttention(channels=nc, downsampling=1, initial_gamma=-8)

        self.head = nn.Sequential(
            AdaptiveConcatPool2d(), Flatten(),
            nn.Linear(2*nc, 512), nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(512, num_classes)
        if in_channels != 3:
            self.encoder[0].conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

    def forward(self, x):
        # x: bs x N x C x W x W
        bs, n, ch, w, h = x.shape
        x = x.view(bs*n, ch, w, h) # x: N bs x C x W x W
        x = self.encoder(x) # x: N bs x C' x W' x W'

        # Concat and pool
        bs2, ch2, w2, h2 = x.shape
        x = x.view(-1, n, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
            .contiguous().view(bs, ch2, n*w2, h2) # x: bs x C' x N W'' x W''

        x, _ = self.self_attention(x)  # x: bs x C' x N W' x W'   bag-wise

        feature_output = self.head(x)

        x = self.fc(feature_output)
        return feature_output, x

    def __repr__(self):
        return f'MIL({self.model_name})'

sigmoid = nn.Sigmoid()
class Swish(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return Swish.apply(x)

class MetaMIL(nn.Module):
    def __init__(self, base_model, num_classes=2, in_channels=3, effnet=False, n_meta_features=None):
        super(MetaMIL, self).__init__()

        self.model_name = base_model.__class__.__name__
        self.effnet = effnet
        self.encoder, nc = drop_fc(base_model)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            AdaptiveConcatPool2d(), Flatten(),
            nn.Linear(2*nc, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 2) # added
        )

        self.meta_nn = nn.Sequential(
            nn.Linear(n_meta_features, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(p=0.7),
            nn.Linear(512, 128),  # FC layer output will have 250 features
            nn.BatchNorm1d(128),
            Swish_Module(),
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(2 + 128, num_classes)
        )


    def forward(self, x, meta):
        # x: bs x N x C x W x W
        bs, n, ch, w, h = x.shape
        x = x.view(bs*n, ch, w, h) # x: N bs x C x W x W
        x = self.encoder(x) # x: N bs x C' x W' x W'

        # Concat and pool
        bs2, ch2, w2, h2 = x.shape
        x = x.view(-1, n, ch2, w2, h2).permute(0, 2, 1, 3, 4)\
            .contiguous().view(bs, ch2, n*w2, h2) # x: bs x C' x N W'' x W''
        feature_output = self.head(x)

        # return feature_output, self.meta_nn(torch.cat((feature_output, meta), dim=1))

        meta = self.meta_nn(meta)
        x = torch.cat((feature_output, meta), dim=1)

        x = self.fc(x)
        return feature_output, x

    def __repr__(self):
        return f'MetaMIL({self.model_name})'

class AttentionMILModel(nn.Module):

    def __init__(self, base_model, num_instances=3,
                 num_classes=2, gated_attention=True):

        super(AttentionMILModel, self).__init__()

        self.model_name = base_model.__class__.__name__
        self.encoder, nc = drop_fc(base_model)
        self.squeeze_flatten = nn.Sequential(
            AdaptiveConcatPool2d(), Flatten())
        self.attention = MultiInstanceAttention(
            2*nc, num_instances, 1, gated_attention=gated_attention)
        self.classifier = nn.Sequential(
            Flatten(), nn.Linear(2*nc, num_classes))

    def forward(self, x):
        bs, n, ch, w, h = x.shape
        x =  x.view(bs*n, ch, w, h) # x: bs N x C x W x W
        x = self.encoder(x)  # x: bs N x C' x W' x W'
        x = self.squeeze_flatten(x)  # x: bs N x C'
        x = x.view(bs, n, -1)  # x: bs x N x C'
        a = self.attention(x)  # a: bs x 1 x N
        # x = torch.matmul(a, x)  # x: bs x 1 x C'
        x = torch.matmul((1+a), x)
        x = self.classifier(x)
        return x, x

    def __repr__(self):
        return f'AMIL({self.model_name})'
