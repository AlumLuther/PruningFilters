import torchvision
import torchvision.transforms as transforms
import torch
from vgg import MyVGG
from resnet import resnet32


class AverageMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_normalizer(data_set, inverse=False):
    if data_set == 'CIFAR10':
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)

    elif data_set == 'CIFAR100':
        MEAN = (0.5071, 0.4867, 0.4408)
        STD = (0.2675, 0.2565, 0.2761)

    else:
        raise RuntimeError("Not expected data flag !!!")

    if inverse:
        MEAN = [-mean / std for mean, std in zip(MEAN, STD)]
        STD = [1 / std for std in STD]

    return transforms.Normalize(MEAN, STD)


def get_transformer(data_set, imsize=None, cropsize=None, crop_padding=None, hflip=None):
    transformers = []
    if imsize:
        transformers.append(transforms.Resize(imsize))
    if cropsize:
        transformers.append(transforms.RandomCrop(cropsize, padding=crop_padding))
    if hflip:
        transformers.append(transforms.RandomHorizontalFlip(hflip))

    transformers.append(transforms.ToTensor())
    transformers.append(get_normalizer(data_set))

    return transforms.Compose(transformers)


def get_data_set(args, train_flag=True):
    if train_flag:
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=True,
                                                                transform=get_transformer(args.data_set, args.imsize,
                                                                                          args.cropsize, args.crop_padding, args.hflip), download=True)
    else:
        data_set = torchvision.datasets.__dict__[args.data_set](root=args.data_path, train=False,
                                                                transform=get_transformer(args.data_set), download=True)
    return data_set


def load_network(args):
    network = None
    if args.load_path:
        check_point = torch.load(args.load_path)
        if args.network == 'vgg':
            network = MyVGG(cfg=check_point['cfg'])
            network.load_state_dict(check_point['state_dict'])
        elif args.network == 'resnet':
            network = resnet32(cfg=check_point['cfg'])
            network.load_state_dict(check_point['state_dict'])
    return network


def save_network(network, args):
    if args.network == 'vgg':
        save_vgg(network, args.save_path)
    elif args.network == 'resnet':
        save_resnet(network, args.save_path)


def save_vgg(network, save_path):
    torch.save({'cfg': get_vgg_cfg(network), 'state_dict': network.state_dict()}, save_path)


def save_resnet(network, save_path):
    torch.save({'cfg': get_resnet_cfg(network), 'state_dict': network.state_dict()}, save_path)


def get_vgg_cfg(network):
    res = []
    for i in range(len(network.features)):
        if isinstance(network.features[i], torch.nn.Conv2d):
            res.append(network.features[i].out_channels)
        elif isinstance(network.features[i], torch.nn.MaxPool2d):
            res.append('M')
    return res


def get_resnet_cfg(network):
    res = [network.stage_1[0].conv_a.out_channels, network.stage_2[0].conv_a.out_channels, network.stage_3[0].conv_a.out_channels]
    return res
