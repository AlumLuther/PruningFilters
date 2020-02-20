import torch

from resnet import *
from softprune import soft_prune_network
from vgg import MyVGG, save_vgg
from parameter import get_parameter
from train import train_network
from evaluate import test_network
from hardprune import hard_prune_network

if __name__ == '__main__':
    network = None
    args = get_parameter()
    if args.load_path:
        check_point = torch.load(args.load_path)
        if args.network == 'vgg':
            network = MyVGG(check_point['cfg'])
            network.load_state_dict(check_point['state_dict'])
        elif args.network == 'resnet':
            network = resnet20()
            network.load_state_dict(check_point['state_dict'])

    if args.train_flag:
        network = train_network(network, args)
    elif args.test_flag:
        network = test_network(network, args)
    elif args.hard_prune_flag:
        network = hard_prune_network(network, args)
    elif args.soft_prune_flag:
        network = soft_prune_network(network, args)

    if args.network == 'vgg':
        save_vgg(network, args.save_path)
    elif args.network == 'resnet':
        save_resnet(network, args.save_path)
