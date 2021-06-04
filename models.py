import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

######### Vanilla CNN model #########
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
class Vanilla_cnn(nn.Module):
    def __init__(self, num_classes=10):
        super(Vanilla_cnn, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=True),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

class Vanilla_cnn_multiclassifier(nn.Module):
    def __init__(self, num_classes=5, num_tasks=20):
        super(Vanilla_cnn_multiclassifier, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier_list = nn.Linear(4096, num_classes * num_tasks)

    def forward(self, x, tid, num_classes=5):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out_final = torch.zeros(out.size(0), num_classes).to(device)
        for i in range(out.shape[0]):
            out_final[i] = torch.matmul(self.classifier_list.weight[num_classes * tid[i]: num_classes * (tid[i]+1), :], out[i]) + self.classifier_list.bias[num_classes * tid[i]: num_classes * (tid[i]+1)]
        return out_final


    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=True),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

######### ResNet-18 #########
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.linear = nn.Linear(128*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=10, skipinit=True):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

class Resnet_multiclassifier(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_tasks=10):
        super(Resnet_multiclassifier, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.linear_list = nn.Linear(128*block.expansion, num_classes * num_tasks)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, tid, num_classes=10):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out_final = torch.zeros(out.size(0), num_classes).to(device)
        for i in range(out.shape[0]):
            out_final[i] = torch.matmul(self.linear_list.weight[num_classes * tid[i]: num_classes * (tid[i]+1), :], out[i]) + self.linear_list.bias[num_classes * tid[i]: num_classes * (tid[i]+1)]
        return out_final

def ResNet18_multiclassifier(num_classes=10, num_tasks=10):
    return Resnet_multiclassifier(BasicBlock, [2,2,2,2], num_classes=num_classes, num_tasks=num_tasks)
