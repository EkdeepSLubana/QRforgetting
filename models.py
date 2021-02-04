import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

######### 6-layer CNN model #########
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']

class Vanilla_cnn(nn.Module):
    '''
    Description:
    A 6-layer CNN that has multiples of 64 filters, like a VGG model.
    '''
    def __init__(self, num_classes=10):
        super(Vanilla_cnn, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(4096, num_classes)
        # self.classifier = nn.Linear(8192, num_classes)

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
    '''
    Description:
    This is redefinition of the above architecture for use with A-GEM and ER-Reservoir.
    It allows indexing into classifier columns, hence the other columns are not affected.
    Note that this terribly slows down training; unsure if it can be made faster, but this def works.
    '''
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