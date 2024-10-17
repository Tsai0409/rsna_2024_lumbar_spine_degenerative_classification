import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as st

class MultiLayerPerceptronBase(nn.Module):
    def __init__(self, num_classes, input_num):
        super(MultiLayerPerceptronBase, self).__init__()
        self.fc = nn.Linear(input_num, num_classes)

    def forward(self, x, labels=None):
        x = self.fc(x)
        return x
class MLP_RSNA1(nn.Module):
    def __init__(self, num_classes, input_num):
        super(MLP_RSNA1, self).__init__()
        self.input_num = input_num
        self.num_chunks = input_num // 150
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(150, num_classes),
            ) for _ in range(self.num_chunks)
        ])

    def forward(self, x, labels=None):
        xs = []
        for i in range(self.num_chunks):
            chunk = x[:, i*150:(i+1)*150]
            xs.append(self.mlps[i](chunk))
        x = torch.stack(xs, dim=1).mean(dim=1)
        return x
class MLP_RSNA2(nn.Module):
    def __init__(self, num_classes, input_num):
        super(MLP_RSNA2, self).__init__()
        self.input_num = input_num
        self.num_chunks = input_num // 150
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(150, num_classes),
            ) for _ in range(self.num_chunks)
        ])

    def forward(self, x, labels=None):
        xs = []
        for i in range(self.num_chunks):
            chunk = x[:, i*150:(i+1)*150]
            xs.append(self.mlps[i](chunk))
        x = torch.stack(xs, dim=1).mean(dim=1)
        return x
class MLP_RSNA3(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA3, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(k), len(v)),
            ) for k, v in in_out_map
        ])

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            o = self.mlps[i](x[:, k])
            output[:, v] = o
        return output
class MLP_RSNA4(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA4, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(k), len(v), bias=False),
            ) for k, v in in_out_map
        ])

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            o = self.mlps[i](x[:, k])
            output[:, v] = o
        return output
class MLP_RSNA7(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA7, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlps = nn.ModuleList()
        self.mlps.append(
            nn.Sequential(
                nn.Linear(5, 5),
                nn.ReLU(),
                nn.Linear(5, 5),
            )
        )
        for n, (k, v) in enumerate(in_out_map):
            if n == 0:
                continue
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(len(k), len(v), bias=False),
                )
            )            
        # st()
    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            o = self.mlps[i](x[:, k])
            output[:, v] = o
        return output
class MLP_RSNA5(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA5, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp = nn.Sequential(
            nn.Linear(input_num, input_num*2),
            nn.ReLU(),
            nn.Linear(input_num*2, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            o = self.mlp(x[:, k])
            output[:, v] = o
        return output
class MLP_RSNA5_v2(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA5_v2, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )

class MLP_RSNA5_v3(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA5_v3, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp = nn.Sequential(
            nn.Linear(input_num, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            o = self.mlp(x[:, k])
            output[:, v] = o
        return output

class MLP_RSNA9(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA9, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp_spinal = nn.Sequential(
            nn.Linear(input_num, input_num*2),
            nn.ReLU(),
            nn.Linear(input_num*2, 3),
        )
        self.mlp_nfn = nn.Sequential(
            nn.Linear(input_num, input_num*2),
            nn.ReLU(),
            nn.Linear(input_num*2, 3),
        )
        self.mlp_ss = nn.Sequential(
            nn.Linear(input_num, input_num*2),
            nn.ReLU(),
            nn.Linear(input_num*2, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            if i in [0,1,2,3,4]:
                o = self.mlp_spinal(x[:, k])
            elif i in [5,6,7,8,9,10,11,12,13,14]:
                o = self.mlp_nfn(x[:, k])
            else:
                o = self.mlp_ss(x[:, k])

            output[:, v] = o
        return output
class MLP_RSNA11(nn.Module):
    def __init__(self, num_classes, in_out_map, input_num=3):
        super(MLP_RSNA11, self).__init__()
        self.in_out_map = in_out_map
        self.mlp_spinal = nn.Sequential(
            nn.Linear(input_num, input_num*2),
            nn.ReLU(),
            nn.Linear(input_num*2, 1),
        )
        self.mlp_nfn = nn.Sequential(
            nn.Linear(input_num, input_num*2),
            nn.ReLU(),
            nn.Linear(input_num*2, 3),
        )
        self.mlp_ss = nn.Sequential(
            nn.Linear(input_num, input_num*2),
            nn.ReLU(),
            nn.Linear(input_num*2, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            if i in range(15):
                o = self.mlp_spinal(x[:, k])
            elif i in range(15, 25):
                o = self.mlp_nfn(x[:, k])
            else:
                o = self.mlp_ss(x[:, k])

            output[:, v] = o
        return output
class MLP_RSNA12(nn.Module):
    def __init__(self, num_classes, in_out_map, input_num=3):
        super(MLP_RSNA12, self).__init__()
        self.in_out_map = in_out_map
        self.mlp_spinal = nn.Sequential(
            nn.Linear(input_num*3, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 1),
        )
        self.mlp_nfn = nn.Sequential(
            nn.Linear(input_num, input_num*2),
            nn.ReLU(),
            nn.Linear(input_num*2, 1),
        )
        self.mlp_ss = nn.Sequential(
            nn.Linear(input_num, input_num*2),
            nn.ReLU(),
            nn.Linear(input_num*2, 1),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            if i in range(15):
                o = self.mlp_spinal(x[:, k])
            elif i in range(15, 25):
                o = self.mlp_nfn(x[:, k])
            else:
                o = self.mlp_ss(x[:, k])

            output[:, v] = o
        return output
class MLP_RSNA8(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA8, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp = nn.Sequential(
            nn.Linear(input_num, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            o = self.mlp(x[:, k])
            output[:, v] = o
        return output
class MLP_RSNA6(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA6, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map

        dropout = 0.3
        self.mlp = nn.Sequential(
            nn.Linear(input_num, 1),
            nn.ReLU(),
            nn.Linear(1, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            o = self.mlp(x[:, k])
            output[:, v] = o
        return output
class MultiLayerPerceptronv2(nn.Module):
    def __init__(self, num_classes, input_num, dropout=0.3):
        super(MultiLayerPerceptronv2, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_num, 36),
            nn.ReLU(),
            nn.Linear(36, num_classes),
        )

    def forward(self, x, labels=None):
        x = self.mlp(x)
        return x
class MultiLayerPerceptronv2NoBias(nn.Module):
    def __init__(self, num_classes, input_num, dropout=0.3):
        super(MultiLayerPerceptronv2NoBias, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_num, 36, bias=False),
            nn.ReLU(),
            nn.Linear(36, num_classes, bias=False),
        )

    def forward(self, x, labels=None):
        x = self.mlp(x)
        return x

class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_classes, input_num, dropout=0.3):
        super(MultiLayerPerceptron, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_num),
            nn.Linear(input_num, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, labels=None):
        x = self.mlp(x)
        return x

class MultiLayerPerceptron3(nn.Module):
    def __init__(self, num_classes, input_num, dropout=0.3):
        super(MultiLayerPerceptron3, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_num),
            nn.Linear(input_num, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, labels=None):
        x = self.mlp(x)
        return x

class MultiLayerPerceptron5(nn.Module):
    def __init__(self, num_classes, input_num, dropout=0.3):
        super(MultiLayerPerceptron5, self).__init__()
        self.bowel = nn.Linear(9, 2)
        self.extravasation = nn.Linear(9, 2)
        self.kidney = nn.Linear(33, 3)
        self.liver = nn.Linear(27, 3)
        self.spleen = nn.Linear(27, 3)

    def forward(self, x, labels=None):
        bowel = self.bowel(x[:, :9])
        extravasation = self.extravasation(x[:, 9:18])
        kidney = self.kidney(x[:, 18:51])
        liver = self.liver(x[:, 51:78])
        spleen = self.spleen(x[:, 78:])
        return torch.cat([bowel, extravasation, kidney, liver, spleen], 1)

class MultiLayerPerceptron4(nn.Module):
    def __init__(self, num_classes, input_num, dropout=0.3):
        super(MultiLayerPerceptron4, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_num),
            nn.Linear(input_num, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, labels=None):
        x = self.mlp(x)
        return x

class MultiLayerPerceptron2(nn.Module):
    def __init__(self, num_classes, input_num, dropout=0.4):
        super(MultiLayerPerceptron2, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_num),
            nn.Linear(input_num, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(512, 1028),
            nn.BatchNorm1d(1028),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(1028, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, labels=None):
        x = self.mlp(x)
        return x

class MLP_RSNA9_v2(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA9_v2, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp_spinal = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )
        self.mlp_nfn = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )
        self.mlp_ss = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            if i in [0,1,2,3,4]:
                o = self.mlp_spinal(x[:, k])
            elif i in [5,6,7,8,9,10,11,12,13,14]:
                o = self.mlp_nfn(x[:, k])
            else:
                o = self.mlp_ss(x[:, k])

            output[:, v] = o
        return output


class MLP_RSNA9_v7(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA9_v7, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp_spinal = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.BatchNorm1d(input_num),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )
        self.mlp_nfn = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.BatchNorm1d(input_num),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )
        self.mlp_ss = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.BatchNorm1d(input_num),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            if i in [0,1,2,3,4]:
                o = self.mlp_spinal(x[:, k])
            elif i in [5,6,7,8,9,10,11,12,13,14]:
                o = self.mlp_nfn(x[:, k])
            else:
                o = self.mlp_ss(x[:, k])

            output[:, v] = o
        return output
class MLP_RSNA9_v2_drop(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA9_v2, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp_spinal = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )
        self.mlp_nfn = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )
        self.mlp_ss = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            if i in [0,1,2,3,4]:
                o = self.mlp_spinal(x[:, k])
            elif i in [5,6,7,8,9,10,11,12,13,14]:
                o = self.mlp_nfn(x[:, k])
            else:
                o = self.mlp_ss(x[:, k])

            output[:, v] = o
        return output
class MLP_RSNA9_v3(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA9_v3, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp_spinal = nn.Sequential(
            nn.Linear(input_num, 3),
        )
        self.mlp_nfn = nn.Sequential(
            nn.Linear(input_num, 3),
        )
        self.mlp_ss = nn.Sequential(
            nn.Linear(input_num, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            if i in [0,1,2,3,4]:
                o = self.mlp_spinal(x[:, k])
            elif i in [5,6,7,8,9,10,11,12,13,14]:
                o = self.mlp_nfn(x[:, k])
            else:
                o = self.mlp_ss(x[:, k])

            output[:, v] = o
        return output
class MLP_RSNA9_v3(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA9_v3, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp_spinal = nn.Sequential(
            nn.Linear(input_num, 3),
        )
        self.mlp_nfn = nn.Sequential(
            nn.Linear(input_num, 3),
        )
        self.mlp_ss = nn.Sequential(
            nn.Linear(input_num, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            if i in [0,1,2,3,4]:
                o = self.mlp_spinal(x[:, k])
            elif i in [5,6,7,8,9,10,11,12,13,14]:
                o = self.mlp_nfn(x[:, k])
            else:
                o = self.mlp_ss(x[:, k])

            output[:, v] = o
        return output
class MLP_RSNA9_v4(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map):
        super(MLP_RSNA9_v4, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp_spinal = nn.Sequential(
            nn.Linear(30, 15),
            nn.BatchNorm1d(15),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(15, 15),
        )
        self.mlp_nfn = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )
        self.mlp_ss = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            if i in [0]:
                o = self.mlp_spinal(x[:, k])
            elif i in [1,2,3,4,5,6,7,8,9,10]:
                o = self.mlp_nfn(x[:, k])
            else:
                o = self.mlp_ss(x[:, k])

            output[:, v] = o
        return output
class MLP_RSNA9_v5(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map, spinal_input_num=30):
        super(MLP_RSNA9_v5, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp_spinal = nn.Sequential(
            nn.Linear(spinal_input_num, 15),
            nn.BatchNorm1d(15),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(15, 15),
        )
        self.mlp_nfn = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )
        self.mlp_ss = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            if i in [0]:
                o = self.mlp_spinal(x[:, k])
            elif i in [1,2,3,4,5,6,7,8,9,10]:
                o = self.mlp_nfn(x[:, k])
            else:
                o = self.mlp_ss(x[:, k])

            output[:, v] = o
        return output
class MLP_RSNA9_v6(nn.Module):
    def __init__(self, num_classes, input_num, in_out_map, spinal_input_num=30):
        super(MLP_RSNA9_v6, self).__init__()
        self.input_num = input_num
        self.in_out_map = in_out_map
        self.mlp_spinal = nn.Sequential(
            nn.Linear(spinal_input_num, 5),
            nn.BatchNorm1d(5),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.BatchNorm1d(5),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(5, 15),
        )

        self.mlp_nfn = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )
        self.mlp_ss = nn.Sequential(
            nn.Linear(input_num, input_num),
            nn.ReLU(),
            nn.Linear(input_num, 3),
        )

    def forward(self, x, labels=None):
        output = torch.zeros(x.size(0), 75, device='cuda')
        for i, (k, v) in enumerate(self.in_out_map):
            if i in [0]:
                o = self.mlp_spinal(x[:, k])
            elif i in [1,2,3,4,5,6,7,8,9,10]:
                o = self.mlp_nfn(x[:, k])
            else:
                o = self.mlp_ss(x[:, k])

            output[:, v] = o
        return output
