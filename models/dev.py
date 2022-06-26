
import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, p=2.0, dim=1, eps=1e-12):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, self.eps)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.activate_before_residual = activate_before_residual

        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
        

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.shortcut(x), out)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, drop_rate=0.0):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, f"depth should be 6*k+4 where k is int"
        n = (depth - 4) // 6
        block = BasicBlock
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]

        self.widen_factor = widen_factor
        self.depth = depth
        self.drop_rate = drop_rate
        self.channels = channels[0]
        # 1st conv before any network block
        self.conv = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        self.block1 = self._make_layer(block, channels[1], n, 1, drop_rate, True)
        self.block2 = self._make_layer(block, channels[2], n, 2, drop_rate, False)
        self.block3 = self._make_layer(block, channels[3], n, 2, drop_rate, False)
        # global average pooling and classifier
        self.bn = nn.BatchNorm2d(self.channels, momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)

        # init_wegihts
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, out_planes, num_layers, stride, drop_rate, activate_before_residual=False):
        layers = []
        for i in range(num_layers):
            layers.append(
                block(i == 0 and self.channels or out_planes, out_planes, i == 0 and stride or 1, drop_rate, activate_before_residual)
            )
        self.channels = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


class Network(nn.Module):
    def __init__(self, backbone, cls_head=None, proj_head=None):
        super(Network, self).__init__()
        self.backbone = backbone
        self.cls_head = nn.Identity() if cls_head is None else cls_head
        self.proj_head = nn.Identity() if proj_head is None else proj_head
    
    def forward(self, x):
        x = self.backbone(x)
        if self.training:
            return self.cls_head(x), self.proj_head(x), x
        else:
            return self.cls_head(x)


class ProjNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout, num_classes):
        super(ProjNet, self).__init__()
        self.backbone = WideResNet(depth, widen_factor, dropout)
        self.pred_head = nn.Linear(self.backbone.channels, num_classes)
        self.proj_head = nn.Sequential(
            nn.Linear(self.backbone.channels, self.backbone.channels),
            nn.ReLU(),
            nn.Linear(self.backbone.channels, self.backbone.channels),
        )
    
    def forward(self, x, proj=False):
        x = self.backbone(x)
        if proj:
            x = self.proj_head(x)
        x = self.pred_head(x)
        return x
        
        
def build(depth, widen_factor, num_classes, dropout=0.0, proj=False, low_dim=64, **kwargs):
    backbone = WideResNet(depth, widen_factor, dropout)
    cls_head = nn.Sequential(
        nn.Linear(backbone.channels, num_classes)
    )
    proj_head = None
    # proj_head = nn.Sequential(
    #     nn.Linear(backbone.channels, backbone.channels),
    #     nn.LeakyReLU(inplace=True, negative_slope=0.1),
    #     nn.Linear(backbone.channels, low_dim),
    #     Normalize(p=2)
    # )
    return Network(backbone, cls_head, proj_head)


if __name__ == "__main__":
    import torch
    cfg = {
        "depth": 28,
        "widen_factor": 2,
        "num_classes": 10,
        "proj": True,
        "proj_after": True
    }
    model = build(**cfg)
    x = torch.rand((4, 3, 32, 32))
    with torch.no_grad():
        outputs = model(x)
        if isinstance(outputs, tuple):
            print([out.shape for out in outputs])
        else:
            print(outputs.shape)