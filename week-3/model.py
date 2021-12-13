import torch
import torch.nn as nn

VGG16_type = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG16(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGG16, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_net = self.create_conv_layer(VGG16_type)
        self.fcs = nn.Sequential(
            nn.Linear(in_features=7*7*512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=num_classes),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        x = self.conv_net(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x
    def create_conv_layer(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(x),
                    nn.ReLU()
                ]
                in_channels = x
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

