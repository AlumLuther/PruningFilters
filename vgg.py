import torch


class MyVGG(torch.nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.features = None
        self.classifier = None
        self.make_layers(cfg)

    def make_layers(self, cfg, batch_norm=True, num_class=10):
        if cfg is None:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        input_channel = 3
        for curLayer in cfg:
            if curLayer == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
                continue
            layers += [torch.nn.Conv2d(input_channel, curLayer, kernel_size=3, padding=1)]
            if batch_norm:
                layers += [torch.nn.BatchNorm2d(curLayer)]
            layers += [torch.nn.ReLU(inplace=True)]
            input_channel = curLayer
        self.features = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(cfg[-2], 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_class))

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output


def get_cfg(network):
    res = []
    for i in range(len(network.features)):
        if isinstance(network.features[i], torch.nn.Conv2d):
            res.append(network.features[i].out_channels)
        elif isinstance(network.features[i], torch.nn.MaxPool2d):
            res.append('M')
    return res


def save_vgg(network, save_path):
    torch.save({'cfg': get_cfg(network), 'state_dict': network.state_dict()}, save_path)
